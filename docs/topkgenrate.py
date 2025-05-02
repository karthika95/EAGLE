# This function performs top-K generation, building a tree-like structure of generated sequences and selecting the top tokens at each step based on their cumulative scores. It uses attention key-value caching for efficiency and applies a tree mask to maintain the structural integrity of the generation process.
def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
    # Function to generate a sequence using the top-K decoding strategy.
    # Args:
    # - hidden_states: The current hidden states of the model.
    # - input_ids: The token IDs of the input sequence.
    # - head: A projection layer for computing logits from hidden states.
    # - logits_processor: A processor for modifying logits during generation.

    input_ids = input_ids.to(hidden_states.device)
    # Ensure that input_ids are on the same device as hidden_states.

    total_tokens = self.total_tokens
    depth = self.depth
    top_k = self.top_k
    # Initialize variables for the total number of tokens, depth of generation, and top_k selection.

    sample_token = input_ids[:, -1]
    # Extract the last token from the input sequence to use as the starting point for generation.

    scores_list = []
    parents_list = []
    ss_token = []
    # Initialize lists to store scores, parent indices (for tree structure), and generated tokens.

    input_ids = input_ids[:, 1:]
    input_ids = input_ids.to(hidden_states.device)
    # Remove the first token from input_ids and ensure it is on the correct device.

    len_posi = input_ids.shape[1]
    self.reset()
    # Calculate the length of the input sequence and reset the tree mask.

    # Check if stable_kv (cached key-value pairs for attention) is available.
    if hasattr(self, "stable_kv") and self.stable_kv is not None:
        kv_len = self.stable_kv[0][0].shape[2]
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:], past_key_values=self.stable_kv, use_cache=True)
    else:
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
    # Pass the input through the model to get initial hidden states and cache past key-value pairs.

    self.stable_kv = past_key_values
    # Cache the past key-value pairs for future use.

    last_hidden = out_hidden[:, -1]
    # Extract the hidden state of the last token.

    last_headout = head(last_hidden)
    # Use the head to compute logits from the last hidden state.

    last_p = self.logsoftmax(last_headout)
    # Apply log-softmax to the logits to obtain probabilities.

    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values
    # Retrieve the top_k token indices and their probabilities.

    scores = topk_p[0]
    scores_list.append(scores[None])
    parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
    ss_token.append(topk_index)
    # Save the top_k scores, initialize parent indices, and store the top_k tokens.

    input_ids = topk_index
    input_hidden = last_hidden[None].repeat(1, top_k, 1)
    # Prepare the input tokens and hidden states for the next step.

    tree_mask = self.tree_mask_init
    topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
    # Initialize the tree mask and indices for top_k tokens.

    # Generate tokens iteratively for the specified depth.
    for i in range(depth):
        self.tree_mask = tree_mask
        position_ids = len_posi + self.position_ids
        # Update the tree mask and position IDs for this step.

        out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values, position_ids=position_ids, use_cache=True)
        len_posi += 1
        # Pass the current input through the model to get updated hidden states and past key-value pairs.

        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k ** 2 * bias2 + bias1
        parents = (topk_cs_index + bias)
        parents_list.append(parents)
        # Compute parent indices for the tree structure.

        last_headout = head(out_hidden[0])
        last_p = self.logsoftmax(last_headout)
        # Compute logits and log-probabilities for the current step.

        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        # Retrieve the top_k token indices and their probabilities for the current step.

        cu_scores = topk_p + scores[:, None]
        # Cumulatively add scores from the previous step.

        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p
        # Select the top_k scores and their indices from the cumulative scores.

        out_ids = topk_cs_index // top_k
        input_hidden = out_hidden[:, out_ids]
        # Update the input hidden states based on the selected indices.

        input_ids = topk_index.view(-1)[topk_cs_index][None]
        # Update the input tokens for the next step.

        ss_token.append(topk_index)
        scores_list.append(cu_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
        # Update the tree mask and store the generated tokens and scores.

    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)
    # Flatten the scores and tokens lists for further processing.

    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    # Select and sort the top scores.

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
    # Retrieve the draft tokens based on the top scores.

    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()
    # Compute the parent indices for the draft tokens.

    tree_mask = torch.eye(total_tokens + 1).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
    # Build the tree mask based on the parent indices.

    tree_position_ids = torch.sum(tree_mask, dim=1) - 1
    # Compute the position IDs for the tree structure.

    tree_mask = tree_mask.float()[None, None]
    draft_tokens = draft_tokens[None]
    # Prepare the final tree mask and draft tokens for output.

    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num
    # Calculate the maximum depth of the tree and the number of leaf and non-leaf nodes.

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()
    # Initialize the indices for retrieving tokens from the tree.

    rid = 0
    position_ids_list = tree_position_ids.tolist()
    # Variables for iterating through the tree structure.

    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = position_ids_list[i]
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1
    # Populate the retrieve indices for each leaf node.

    if logits_processor is not None:
        maxitem = total_tokens + 5

        def custom_sort(lst):
            # Sort the list based on token values, treating negative values as maximum.
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    # Apply a custom sorting function to the retrieve indices if logits_processor is provided.

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
    tree_position_ids = tree_position_ids.to(hidden_states.device)
    # Finalize retrieve indices and clean up intermediate variables.

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
    # Return the generated tokens, retrieve indices, tree mask, and position IDs.