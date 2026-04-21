from typing import List
from samd.sam.dyn_sam import DynSAM


class WordGroupAwareDynSAM(DynSAM):

    def __init__(self, n_predicts: int = 40):
        super().__init__(n_predicts)
        self.word_boundaries: List[bool] = [False]

    def reset(self):
        super().reset()
        self.word_boundaries = [False]


    def add_tokens(self, tokens: List[int], boundaries: List[bool] = None):

        if boundaries is None:
            boundaries = [False] * len(tokens)

        for token, is_boundary in zip(tokens, boundaries):
            self.transfer_cur_state(token)
            self.add_state(token)           
            self.word_boundaries.append(is_boundary)

        self.input_ids.extend(tokens)

    def transfer_tokens(self, tokens: List[int], boundaries: List[bool] = None):

        if boundaries is None:
            boundaries = [False] * len(tokens)

        for token, _ in zip(tokens, boundaries):
            self.transfer_cur_state(token)

    def gen_draft(self, index: int, start_token: int):


        if index == 0:
            return [start_token] + [0] * (self.n_predicts - 1), 0

        endpos = self.states[index].min_endpos

    # Start after the match
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

        if last_boundary_offset is not None:
            buffer_tokens = buffer_tokens[: last_boundary_offset + 1]

        pred_ids.extend(buffer_tokens)

        while len(pred_ids) < self.n_predicts:
            pred_ids.append(0)


        return pred_ids, len(buffer_tokens)
