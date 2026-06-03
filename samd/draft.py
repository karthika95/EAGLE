import torch
from typing import List, Tuple, Dict, Optional
from enum import Enum
from collections import namedtuple

from .samd_config import SamdConfig
from .sam import DynSAM, StaticSAM, NullStaticSAM
from .tree_model import TreeModel, tree_model_cls
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer

from profile_utils import profile_decorator, profile_lookup_decorator


class CandidateType(str, Enum):
    sequence = "sequence"
    tree = "tree"

Candidates = namedtuple('Candidates', ['type','seqtype', 'tokens', 'candidate_tokens', 'buffers_kwargs'])
DraftInfo = namedtuple('DraftInfo', ['candidate_type', 'seqtype', 'tokens', 'buffers_kwargs', 'n_draft', 'draft_stats'])

TOPK = 8

class DraftModel(torch.nn.Module):
    
    def __init__(self,
        config: SamdConfig,
        sam_dyn: DynSAM = None,
        sam_static: StaticSAM = None,
        tree_model: TreeModel = None,
        lm: LlamaForCausalLM = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        tree_cls = tree_model_cls[config.tree_method]
        self.config = config
        self.sam_dyn = sam_dyn if sam_dyn is not None else DynSAM(config.n_predicts)
        self.sam_static = sam_static if sam_static is not None else NullStaticSAM(config.n_predicts)
        self.tree_model = tree_model if tree_model is not None else tree_cls(config, lm, dtype, device)
        

        self.sam_dyn.n_predicts = config.n_predicts
        self.sam_static.n_predicts = config.n_predicts
        self.len_bias = config.len_bias
        self.len_threshold = config.len_threshold
        
    def reset(self):
        self.sam_dyn.reset()
        self.sam_static.reset()
        self.tree_model.reset()

    def lookup(self, start_token: int,step):
        counter=0
        index_dyn, match_dyn,counter = self.sam_dyn.lookup(start_token,step,counter)#retrives the index and the length
        index_static, match_static,counter = self.sam_static.lookup(start_token,step,counter)

        match_static -= self.len_bias
        #The match length of the static and dynamic are being compared 
        if max(match_dyn, match_static) >= self.len_threshold:
            if match_dyn >= match_static:
                seq = self.sam_dyn.gen_draft(index_dyn, start_token)
                seqtype = "dynamic"
            else:
                seq = self.sam_static.gen_draft(index_static, start_token)
                seqtype = "static"

            # Normalize sequence return values: some SAM implementations
            # return (pred_ids, meta) while others return pred_ids directly.
            seq_tokens = seq[0] if isinstance(seq, tuple) and len(seq) >= 1 else seq

            if isinstance(seq_tokens, (list, tuple)):
                n_draft = max(0, len(seq_tokens) - 1)
            elif isinstance(seq_tokens, torch.Tensor):
                n_draft = max(0, seq_tokens.shape[-1] - 1)
            else:
                # Fallback: assume single-token draft
                n_draft = 1

            return (CandidateType.sequence, seqtype, seq_tokens, {}, n_draft)
        seqtype = "tree"
        tree_tokens, buffers_kwargs = self.tree_model.gen_draft(start_token)
        n_draft = max(0, len(tree_tokens) - 1)
        return (CandidateType.tree, seqtype, tree_tokens, buffers_kwargs, n_draft)
    
    def lookup_with_stats(self, start_token: int, step: int) -> DraftInfo:
        """Enhanced lookup that returns draft statistics"""
        counter = 0
        index_dyn, match_dyn, counter = self.sam_dyn.lookup(start_token, step, counter)
        index_static, match_static, counter = self.sam_static.lookup(start_token, step, counter)
        
        match_static_adj = match_static - self.len_bias
        draft_stats = {
            "static_match_length": match_static,
            "dynamic_match_length": match_dyn,
            "static_match_length_adjusted": match_static_adj,
        }
        
        # Decision logic: which method to use?
        if max(match_dyn, match_static_adj) >= self.len_threshold:
            if match_dyn >= match_static_adj:
                seq = self.sam_dyn.gen_draft(index_dyn, start_token)
                seqtype = "dynamic"
                draft_stats["selected_method"] = "dynamic"
            else:
                seq = self.sam_static.gen_draft(index_static, start_token)
                seqtype = "static"
                draft_stats["selected_method"] = "static"

            seq_tokens = seq[0] if isinstance(seq, tuple) and len(seq) >= 1 else seq

            if isinstance(seq_tokens, (list, tuple)):
                n_draft = max(0, len(seq_tokens) - 1)
            elif isinstance(seq_tokens, torch.Tensor):
                n_draft = max(0, seq_tokens.shape[-1] - 1)
            else:
                n_draft = 1

            draft_stats["drafted_count"] = n_draft
            return DraftInfo(
                candidate_type=CandidateType.sequence,
                seqtype=seqtype,
                tokens=seq_tokens,
                buffers_kwargs={},
                n_draft=n_draft,
                draft_stats=draft_stats
            )
        
        # Fall back to tree
        seqtype = "tree"
        tree_tokens, buffers_kwargs = self.tree_model.gen_draft(start_token)
        n_draft = max(0, len(tree_tokens) - 1)
        draft_stats["selected_method"] = "tree"
        draft_stats["drafted_count"] = n_draft
        return DraftInfo(
            candidate_type=CandidateType.tree,
            seqtype=seqtype,
            tokens=tree_tokens,
            buffers_kwargs=buffers_kwargs,
            n_draft=n_draft,
            draft_stats=draft_stats
        )
    
    def update(self,
        tokens: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
        tree_tokens: Optional[torch.Tensor] = None,
        tree_logits: Optional[torch.Tensor] = None,
    ):
        tokens_list = tokens.tolist()
        self.sam_dyn.add_tokens(tokens_list)
        self.sam_static.transfer_tokens(tokens_list)

        self.tree_model.update(
            tokens=tokens,
            last_hidden_states=last_hidden_states,
            tree_tokens=tree_tokens, 
            tree_logits=tree_logits,
        )