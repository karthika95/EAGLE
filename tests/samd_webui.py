import argparse
import pickle
import gradio as gr 
import torch
import re
import time
import os
import re

from fastchat.model import get_conversation_template
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    GenerationMixin,
    LlamaConfig,
    LlamaTokenizer
)
import collections
import gc
import webbrowser
from samd import (
    SamdConfig, 
    SamdModel, 
    SamdGenerationConfig,
    DraftModel,
    load_sam
)

from samd.draft import CandidateType
from samd.wordgroup_sam import WordGroupAwareSAM
from samd.wordgroup.grouping import boundaries_for_token_ids

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--sam_path', type=str, default="")
parser.add_argument('--wordgroup_sam', action='store_true', help='Treat sam_path as a word-group-aware SAM pickle')
parser.add_argument('--samd_n_predicts', type=int, default=10)
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--max_cache_len', type=int, default=2048)
parser.add_argument("--tree_method", type=str, default="eagle2")
parser.add_argument("--tree_model_path", type=str, default="")
parser.add_argument('--len_threshold', type=int, default=3)
parser.add_argument('--len_bias', type=int, default=2)
parser.add_argument('--disable_dyn', action='store_true', help='Disable dynamic SAM drafting')
parser.add_argument('--disable_static', action='store_true', help='Disable static SAM drafting')
parser.add_argument('--disable_eagle', action='store_true', help='Disable tree/EAGLE fallback')
parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])
args = parser.parse_args()

args.dtype = {
    'float16': torch.float16,
    'float32': torch.float32,
}[args.dtype]

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path, 
    torch_dtype=args.dtype, 
    device_map=args.device,
)
model.eval()

def load_wordgroup_sam(path: str):
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            sam = pickle.load(f)
        if not isinstance(sam, WordGroupAwareSAM):
            print("[WARN] Loaded SAM is not WordGroupAwareSAM; proceeding anyway.")
        return sam
    except FileNotFoundError:
        print(f"[WARN] wordgroup SAM not found at {path}; continuing without static SAM")
        return None



class ShardedWordGroupSAM:

    def __init__(self, shard_dir):

        self.shard_dir = shard_dir

        meta_path = os.path.join(shard_dir, "sam_meta.pt")
        meta = torch.load(meta_path, map_location="cpu", mmap=False, weights_only=False)

        self.n_predicts = int(meta["n_predicts"])
        self.n_states = int(meta["num_states"])
        self.shard_size = int(meta["shard_size"])

        self.sam = WordGroupAwareSAM(n_predicts=self.n_predicts)

        self.sam.input_ids = meta["input_ids"]
        self.sam.word_boundaries = meta["word_boundaries"]
        self.sam.last = int(meta["last"])
        self.sam.max_length = int(meta["max_length"])


        self.sam.get_state = self.get_state
        self.sam.n_states = self.n_states

        self.shards = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if re.match(r"sam_states_\d+\.pt", f)
        ])

        self.loaded_shard_id = None
        self.loaded = None
        
        self.cache = collections.OrderedDict()
        self.cache_size = 200

    def _load_shard(self, shard_id):

        if shard_id in self.cache:
            self.cache.move_to_end(shard_id)
            self.loaded = self.cache[shard_id]
            self.loaded_shard_id = shard_id
            return

        print(f"Loading shard {shard_id}")
        obj = torch.load(
            self.shards[shard_id],
            map_location="cpu",
            mmap=True,
            weights_only=False
        )

        self.loaded = obj
        self.loaded_shard_id = shard_id
        
        self.cache[shard_id] = obj
        
        if len(self.cache) > self.cache_size:
            oldest_id, _ = self.cache.popitem(last=False)
            print(f"Evicted shard {oldest_id} from cache")
            gc.collect() 

    def get_state(self, index):

        shard_id = index // self.shard_size
        offset   = index % self.shard_size

        self._load_shard(shard_id)

        return WordGroupAwareSAM.SAMState(
            next=self.loaded["transitions"][offset],
            link=int(self.loaded["links"][offset]),
            length=int(self.loaded["lengths"][offset]),
            min_endpos=int(self.loaded["min_endpos"][offset]),
        )


class WordGroupAwareDraftModel(DraftModel):

    def __init__(
        self,
        config,
        sam_static=None,
        lm=None,
        dtype=None,
        device=None,
        tokenizer=None,
        disable_dyn: bool = False,
        disable_eagle: bool = False,
        disable_static: bool = False,
    ):
        from samd.sam.wordgroup_dyn_sam import WordGroupAwareDynSAM

        sam_dyn = WordGroupAwareDynSAM(config.n_predicts)

        self.disable_dyn = disable_dyn
        self.disable_eagle = disable_eagle
        self.disable_static = disable_static
        self.tokenizer = tokenizer

        super().__init__(
            config=config,
            sam_dyn=sam_dyn,
            sam_static=sam_static,
            lm=lm,
            dtype=dtype,
            device=device,
        )

    def update(self, tokens=None, last_hidden_states=None, tree_tokens=None, tree_logits=None):
        if tokens is not None:
            token_list = tokens.tolist()

            text = self.tokenizer.decode(token_list, skip_special_tokens=True)

            boundaries = boundaries_for_token_ids(
                self.tokenizer,
                text,
                token_list
            )

            if not self.disable_dyn:
                self.sam_dyn.add_tokens(token_list, boundaries)

            if self.sam_static is not None and not self.disable_static:
                self.sam_static.transfer_tokens(token_list)

        if not self.disable_eagle:
            self.tree_model.update(
                tokens=tokens,
                last_hidden_states=last_hidden_states,
                tree_tokens=tree_tokens,
                tree_logits=tree_logits,
            )


    def lookup(self, start_token: int, step: int = 0):
        if self.disable_dyn and self.disable_static and self.disable_eagle:
            seq = [start_token] + [0] * (self.sam_dyn.n_predicts - 1)
            return (CandidateType.sequence, "baseline", seq, {}, 0)

        counter = 0
        if not self.disable_dyn:
            index_dyn, match_dyn, counter = self.sam_dyn.lookup(start_token, step, counter)
        else:
            index_dyn, match_dyn = -1, float('-inf')

        if not self.disable_static and self.sam_static is not None:
            index_static, match_static, counter = self.sam_static.lookup(start_token, step, counter)
            match_static -= self.len_bias
        else:
            index_static, match_static = -1, float('-inf')

        best_match = max(match_dyn, match_static)
        # print("The length threshold is.........................................",self.len_threshold)
        threshold_met = best_match >= self.len_threshold

        if threshold_met or self.disable_eagle:
            use_dynamic = (not self.disable_dyn) and (match_dyn >= match_static)
            if use_dynamic:
                seq, n_draft = self.sam_dyn.gen_draft(index_dyn, start_token)
                seqtype = "dynamic"
            else:
                if (not self.disable_static) and (self.sam_static is not None):
                    seq, n_draft = self.sam_static.gen_draft(index_static, start_token)
                    seqtype = "static"
                else:
                    seq = [start_token] + [0] * (self.sam_dyn.n_predicts - 1)
                    n_draft = 0
                    seqtype = "baseline"
            return (CandidateType.sequence, seqtype, seq, {}, n_draft)

        tree_tokens, buffers_kwargs = self.tree_model.gen_draft(start_token)
        n_draft = len(tree_tokens) - 1
        return (CandidateType.tree, "tree", tree_tokens, buffers_kwargs, n_draft)

print("Loading SAM and building SamdModel (one-time setup)...")

if args.disable_static:
    sam = None
elif args.wordgroup_sam:
    sam = load_wordgroup_sam(args.sam_path)
else:
    sam = load_sam(args.sam_path) if args.sam_path is not None else None

samd_config = SamdConfig(
    n_predicts=args.samd_n_predicts,
    tree_method=args.tree_method,
    tree_model_path=args.tree_model_path,
    len_threshold=args.len_threshold,
    len_bias=args.len_bias,
)

if args.wordgroup_sam:
    draft = WordGroupAwareDraftModel(
        samd_config,
        sam_static=sam,
        lm=model,
        dtype=args.dtype,
        device=args.device,
        tokenizer=tokenizer,
        disable_dyn=args.disable_dyn,
        disable_eagle=args.disable_eagle,
        disable_static=args.disable_static,
    )
else:
    draft = DraftModel(
        samd_config,
        sam_static=sam,
        lm=model,
        dtype=args.dtype,
        device=args.device,
    )

samd_model = SamdModel(
    samd_config,
    model,
    draft,
    tokenizer.eos_token_id,
    args.dtype,
    args.device,
    tokenizer=tokenizer,
)
samd_model.eval()

gen_config = SamdGenerationConfig(
    max_new_tokens=args.max_new_tokens,
    max_cache_len=args.max_cache_len,
)

print("SAM and SamdModel ready.")

@torch.inference_mode()
def samd_generate(args, inputs, model, tokenizer):
    assert inputs.input_ids.shape[-1] + args.max_new_tokens <= args.max_cache_len
    gen = samd_model.stream_generate(**inputs, generation_config=gen_config)
    return gen
        

def user(current_text,chatbot,session_state):
    if chatbot is None:
        chatbot=[]
    pure_history=session_state.get("pure_history",[])
    pure_history+=[[current_text,None]]
    session_state["pure_history"]=pure_history
    return "",chatbot+[[current_text,None]],session_state

def clear(history,session):
    pure_history=[]
    session["pure_history"]=pure_history
    history=pure_history
    return history,session

def regenerate(history,session_state):
    if history is None:
        history=[]
    pure_history=session_state.get("pure_history",[])
    pure_history[-1][-1]=None
    session_state["pure_history"]=pure_history
    history[-1][-1]=None
    return history,session_state
    

def bot(chatbot, session_state, disable_dyn_ui, disable_static_ui, disable_eagle_ui):
    draft = samd_model.draft
    if hasattr(draft, "disable_dyn"):
        draft.disable_dyn = bool(disable_dyn_ui)
    if hasattr(draft, "disable_static"):
        draft.disable_static = bool(disable_static_ui)
    if hasattr(draft, "disable_eagle"):
        draft.disable_eagle = bool(disable_eagle_ui)

    pure_history=session_state.get("pure_history",[])
    input_text=pure_history[-1][0]
    inputs = tokenizer(
    input_text, 
    padding=True, 
    return_tensors="pt"
    ).to(args.device)
    
    gen=samd_generate(args, inputs, model, tokenizer)
    coloured_response = ""
    raw_response=""
    accumulated_ids = []
    prev_full_text = ""
    rendered_segments = []
    colour_data={"tree":"green",
                 "static":"orange",
                 "dynamic":"red"
    }

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    method_proposed = {"tree": 0, "static": 0, "dynamic": 0}
    method_accepted = {"tree": 0, "static": 0, "dynamic": 0}

    for chunk in gen:
        token_ids = chunk["ids"]
        seqtype = chunk["seqtype"]
        n_draft = chunk.get("n_draft", 0)
        # print("The ids are",chunk["ids"])
        # print("The sequence type is",chunk["seqtype"])
        color = colour_data.get(seqtype, "white")
        display_color = "white" if seqtype == "baseline" else color

        total_steps += 1
        total_tokens += len(token_ids)
        n_accepted = max(0, len(token_ids) - 1)
        if seqtype in method_proposed:
            method_proposed[seqtype] += n_draft
            method_accepted[seqtype] += n_accepted

        accumulated_ids.extend(token_ids)
        full_text = tokenizer.decode(
            accumulated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if full_text.startswith(prev_full_text):
            chunk_text = full_text[len(prev_full_text):]
        else:
            chunk_text = full_text
            rendered_segments = []
        prev_full_text = full_text
        raw_response = full_text

        if chunk_text:
            rendered_segments.append((chunk_text, display_color))

        coloured_response = "".join(
            f"<span style='color:{seg_color}'>{seg_text}</span>"
            for seg_text, seg_color in rendered_segments
        )
        chatbot[-1][1]=coloured_response

        elapsed_time = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        speed_info = (
            f"**Speed:** {tokens_per_sec:.2f} tokens/sec | "
            f"**Tokens:** {total_tokens} | "
            f"**Steps:** {total_steps}"
        )

        yield chatbot, session_state, speed_info

    pure_history[-1][1] = raw_response
    session_state["pure_history"] = pure_history

    elapsed_time = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0

    method_parts = []
    for m in ("static", "dynamic", "tree"):
        proposed = method_proposed[m]
        accepted = method_accepted[m]
        if proposed > 0:
            rate = accepted / proposed * 100
            method_parts.append(f"**{m}:** {accepted}/{proposed} ({rate:.1f}%)")
    acceptance_info = " | ".join(method_parts) if method_parts else "N/A"

    speed_info = (
        f"**Speed:** {tokens_per_sec:.2f} tokens/sec | "
        f"**Tokens:** {total_tokens} | "
        f"**Steps:** {total_steps} | "
        f"**Time:** {elapsed_time:.2f}s\n\n"
        f"**Acceptance rate** — {acceptance_info}"
    )

    yield chatbot, session_state, speed_info



custom_css="""
#regenerate_button,
#clear_button,
#stop_button,
#send_button {
    background-color: #FFFFFF;
    color: black;
    font-size: 20px;
    border-radius: 8px;
    padding: 10px 25px;
    font-weight: bold;
}
#regenerate_button:hover,
#clear_button:hover,
#stop_button:hover,
#send_button:hover {
    background-color: #FFA500;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    <h1 style="text-align:center; color: orange;">SAM-CHATBOT</h1>
    """)
    gs=gr.State({"pure_history": []})
    chatbot=gr.Chatbot(height=600, show_label=False)
    msg=gr.Textbox(label="Input")
    
    speed_display = gr.Markdown(value="**Speed:** -- tokens/sec", label="Generation Speed")
    with gr.Row():
        disable_dyn_checkbox = gr.Checkbox(label="Disable Dynamic", value=args.disable_dyn)
        disable_static_checkbox = gr.Checkbox(label="Disable Static", value=args.disable_static)
        disable_eagle_checkbox = gr.Checkbox(label="Disable Eagle", value=args.disable_eagle)
    
    with gr.Row():
        send_button=gr.Button("Send",elem_id="send_button")
        stop_button=gr.Button("Stop",elem_id="stop_button")
        regenerate_button=gr.Button("Regenerate",elem_id="regenerate_button")
        clear_button=gr.Button("Clear",elem_id="clear_button")
    with gr.Row():
            gr.Markdown("""
        <h3 style="text-align:center; color: white;">⚪-Verifier 🟢-EAGLE  🟠-Static 🔴-Dynamic</h3>
        """)
    enter_event=msg.submit(user,[msg,chatbot,gs],[msg,chatbot,gs]).then(
        bot,
        [chatbot, gs, disable_dyn_checkbox, disable_static_checkbox, disable_eagle_checkbox],
        [chatbot, gs, speed_display]
    )
    send_event=send_button.click(user,[msg,chatbot,gs],[msg,chatbot,gs]).then(
        bot,
        [chatbot, gs, disable_dyn_checkbox, disable_static_checkbox, disable_eagle_checkbox],
        [chatbot, gs, speed_display]
    )
    clear_event=clear_button.click(clear,[chatbot,gs],[chatbot,gs],cancels=[enter_event,send_event])
    regenerate_event=regenerate_button.click(regenerate,[chatbot,gs],[chatbot,gs]).then(
        bot,
        [chatbot, gs, disable_dyn_checkbox, disable_static_checkbox, disable_eagle_checkbox],
        [chatbot, gs, speed_display]
    )
    stop_event=stop_button.click(None,None,None,cancels=[send_event,enter_event])


demo.launch(server_name="0.0.0.0", server_port=5500, share=True)
