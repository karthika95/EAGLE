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
                seqtype="dynamic"
            else:
                seq = self.sam_static.gen_draft(index_static, start_token)
                seqtype="static"
            return (CandidateType.sequence,seqtype, seq, {})
        else:
            seqtype="tree"
            return (CandidateType.tree,seqtype) + self.tree_model.gen_draft(start_token)
    
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