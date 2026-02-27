
from typing import List
from dataclasses import dataclass
from copy import deepcopy
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samd.sam.static_sam import StaticSAM


class WordGroupAwareSAM(StaticSAM):

    def __init__(self, n_predicts: int = 40):
        super().__init__(n_predicts)
    
        self.word_boundaries: List[bool] = [False] 
        
    def add_batch_tokens_with_boundaries(
        self, 
        batch_data: List[dict], 
        eos_token: int, 
        verbose: bool = True
    ):

        for data in tqdm(batch_data, desc="Building word-group-aware SAM...", disable=not verbose):
            tokens = data['tokens']
            boundaries = data['word_group_boundaries']
            
            if len(tokens) != len(boundaries):
                # print(f"Warning: Token length {len(tokens)} != boundary length {len(boundaries)}, skipping")
                continue
                
            self.add_tokens_with_boundaries(tokens, boundaries)
            
            if tokens[-1] != eos_token:
                self.add_tokens_with_boundaries([eos_token], [True])  
    
    def add_tokens_with_boundaries(self, tokens: List[int], boundaries: List[bool]):
        for token, is_boundary in zip(tokens, boundaries):
            self.transfer_cur_state(token)
            self.add_state(token)
            self.word_boundaries.append(is_boundary)
        self.input_ids.extend(tokens)
    
    def gen_draft(self, index: int, start_token: int):

        # print(f"[STATIC SAM] Generating draft...")
        # print(":" * 80)

        if index == 0:
            # print(f"[STATIC SAM] No match found, returning empty draft")
            return [start_token] + [0] * (self.n_predicts - 1)

        endpos = self.get_state(index).min_endpos

        start_pos = endpos + 1
        pred_ids = [start_token]

        buffer_tokens = []
        current_pos = start_pos

        while len(buffer_tokens) < (self.n_predicts - 1) and current_pos < len(self.input_ids):
            buffer_tokens.append(self.input_ids[current_pos])
            current_pos += 1


        last_boundary_offset = None

        for offset in range(len(buffer_tokens)):
            pos = start_pos + offset
            if pos < len(self.word_boundaries) and self.word_boundaries[pos]:
                last_boundary_offset = offset
        #Cut of the draft at the last word boundary
        if last_boundary_offset is not None:
            buffer_tokens = buffer_tokens[: last_boundary_offset + 1]

        pred_ids.extend(buffer_tokens)

        while len(pred_ids) < self.n_predicts:
            pred_ids.append(0)

        # print(
        #     f"[STATIC SAM] Draft: "
        #     f"{len([t for t in pred_ids if t != 0])} tokens "
        #     f"(first 5: {pred_ids[:5]})"
        # )

        return pred_ids

    @staticmethod
    def build(
        batch_data: List[dict],
        eos_token: int,
        n_predicts: int = 40,
        verbose: bool = True
    ):
        sam = WordGroupAwareSAM(n_predicts)
        sam.add_batch_tokens_with_boundaries(batch_data, eos_token, verbose)
        return sam
