from typing import List, Tuple
import sys

word_group_count=0
total_tokens=0

RULE1_PHRASES = [
    ["दे", "दिया"],
    ["मिला", "दें"],
    ["मुकर", "जाएं"],
    ["सम्मिलित", "करना"],
    ["हाल", "ही", "में"],
    ["कर", "दी"],
    ["दे", "दो"],
    ["दे", "दें"],
    ["दी", "थी"]
]


ATTACH_TO_LEFT = {
    "से", "में", "का", "के", "की", "को", "पर", "ने", "भी", "ही",
    "द्वारा", "वाला", "वाली", "वाले", "जी", "सी", "तरह",
    "दी_थी",  "ईस्वी", "ई.पू."# in case this already appears as such in the data
}

RULE3_MULTIWORDS = [
    ["रहे", "हैं"],
    ["रहा", "है"],
    ["रही", "है"],
    ["सकता", "है"],
    ["सकती", "है"],
    ["सकते", "हैं"],
    ["हो", "गयी"],
    ["हो", "गया"],
    ["के", "लिए"],
    ["के", "बाद"],
    ["के", "साथ"],
    ["के", "बीच"],
    ["के", "दौरान"],
    ["के", "खिलाफ़"],
    ["के", "प्रति"],
    ["की", "ओर"],
    ["बारे", "में"],
    ["के", "मुताबिक़"],
    ["के", "मुताबिक"],
    ["के", "तहत"],
    ["ओर", "से"],
    ["के", "कारण"],
    ["ने", "भी"],
    ["में", "ही"],
    ["ही", "में"],
]

ATTACH_MULTI_TO_LEFT = {"##".join(p) for p in RULE3_MULTIWORDS}


A_ENDING = "ा"
EE_ENDING = "ी"
E_ENDING = "े"

AUX_AFTER_EE = {"जाती","गई", "जाएगी", "जायेगी", "है", "हैं", "थी", "थे", "था"}
AUX_AFTER_A = {"जाता","गया", "जाएगा", "जायेगा","है", "हैं", "थी", "थे"}


def is_number_token(w: str) -> bool:
    """
    Returns True if the token looks like a number.
    Handles:
    - Western digits with commas/decimals: 80,000  3.14
    - Hindi digits: ५, १०, १०००, etc. 
    """
    # Remove common number formatting chars
    cleaned = w.replace(",", "").replace(".", "")
    return cleaned.isdigit()

# def apply_phrase_grouping(tokens: List[str]) -> List[str]:
#     out = []
#     i = 0
#     n = len(tokens)

#     all_phrases = RULE1_PHRASES + RULE3_MULTIWORDS
#     all_phrases = sorted(all_phrases, key=len, reverse=True)

#     while i < n:
#         matched = False
#         for phrase in all_phrases:
#             L = len(phrase)
#             if i + L <= n and tokens[i:i + L] == phrase:
#                 out.append("##".join(phrase))
#                 i += L
#                 matched = True
#                 break
#         if matched:
#             continue

#         w = tokens[i]

#         if w.endswith(EE_ENDING) and i+1<n and tokens[i+1] in AUX_AFTER_EE:
#             out.append(w+"##"+tokens[i+1]); i+=2; continue

#         if w.endswith(A_ENDING) and i+1<n and tokens[i+1] in AUX_AFTER_A:
#             out.append(w+"##"+tokens[i+1]); i+=2; continue

#         if i+1<n and tokens[i+1]=="चाहिए" and (w.endswith(A_ENDING) or w.endswith(EE_ENDING)):
#             out.append(w+"##"+tokens[i+1]); i+=2; continue

#         if w.endswith(E_ENDING) and i+2<n and tokens[i+1] in {"लगता","लगती"} and tokens[i+2]=="है":
#             out.append(w+"##"+tokens[i+1]+"##"+tokens[i+2]); i+=3; continue

#         out.append(w)
#         i += 1

#     return out
def apply_rule5_once(tokens):
    out = []
    i = 0
    n = len(tokens)

    while i < n:
        w = tokens[i]

        # EE ending + auxiliary
        if w.endswith(EE_ENDING) and i + 1 < n and tokens[i + 1] in AUX_AFTER_EE:
            out.append(w + "_" + tokens[i + 1])
            i += 2
            continue

        # A ending + auxiliary
        if w.endswith(A_ENDING) and i + 1 < n and tokens[i + 1] in AUX_AFTER_A:
            out.append(w + "_" + tokens[i + 1])
            i += 2
            continue

        # चाहिए
        if i + 1 < n and tokens[i + 1] == "चाहिए":
            if w.endswith(A_ENDING) or w.endswith(EE_ENDING):
                out.append(w + "_चाहिए")
                i += 2
                continue

        # लगता/लगती है
        if w.endswith(E_ENDING) and i + 2 < n and tokens[i + 1] in {"लगता", "लगती"} and tokens[i + 2] == "है":
            out.append(w + "_" + tokens[i + 1] + "_" + tokens[i + 2])
            i += 3
            continue

        out.append(w)
        i += 1

    return out

def apply_rule5_iterative(tokens):
    while True:
        new_tokens = apply_rule5_once(tokens)
        if new_tokens == tokens:
            break
        tokens = new_tokens
    return tokens

def apply_phrase_grouping(tokens):
    all_phrases = RULE1_PHRASES + RULE3_MULTIWORDS
    all_phrases = sorted(all_phrases, key=len, reverse=True)

    out = []
    i = 0
    n = len(tokens)

    while i < n:
        matched = False
        for phrase in all_phrases:
            L = len(phrase)
            if i + L <= n and tokens[i:i + L] == phrase:
                out.append("_".join(phrase))
                i += L
                matched = True
                break
        if matched:
            continue

        out.append(tokens[i])
        i += 1

    # 🔥 Apply Rule 5 iteratively here
    out = apply_rule5_iterative(out)

    return out

def apply_right_attachment(tokens):
    """
    Rule 4
    - "नहीं" and numbers should be grouped with the word to their right.
    Examples: "नहीं करना" -> "नहीं_करना", "10 ग्राम" -> "10_ग्राम"
    """
    out = []
    i = 0
    n = len(tokens)

    while i < n:
        w = tokens[i]
        # If "नहीं" or numeric token and there is a right neighbor
        if (w == "नहीं" or is_number_token(w)) and i + 1 < n:
            out.append(w + "_" + tokens[i + 1])
            i += 2
        else:
            out.append(w)
            i += 1

    return out


def apply_left_attachment(tokens):
    """
    Rule 2 + left-attachment part of Rule 3
    - Tokens in ATTACH_TO_LEFT attach to the previous token.
    - Tokens in ATTACH_MULTI_TO_LEFT (e.g., "के_लिए") also attach to the previous token.
    """
    out = []
    for tok in tokens:
        if tok in ATTACH_TO_LEFT or tok in ATTACH_MULTI_TO_LEFT:
            if out:
                out[-1] = out[-1] + "_" + tok
            else:
                # No left word; just keep as is
                out.append(tok)
        else:
            out.append(tok)
    return out



def group_sentence(sentence):
    #Apply all grouping rules to one sentence (string).
    
    tokens = sentence.strip().split()
    if not tokens:
        return ""

    tokens = apply_phrase_grouping(tokens)
    tokens = apply_right_attachment(tokens)
    tokens = apply_left_attachment(tokens)

    return " ".join(tokens)


def word_groups_to_token_boundaries(tokenizer, cleaned_text: str, tokens: List[int]):
    boundaries = [False] * len(tokens)
    words = cleaned_text.strip().split()
    word_groups = group_sentence(words)
    if not word_groups or not tokens:
        return boundaries
    
    try:
        full_encoding = tokenizer(cleaned_text, return_offsets_mapping=True, add_special_tokens=True)
        
        if 'offset_mapping' in full_encoding and len(full_encoding['input_ids']) == len(tokens):
            return create_boundaries_with_offsets(tokens, tokenizer, word_groups, cleaned_text, full_encoding['offset_mapping'])
        else:
            return create_boundaries_word_based(tokens, tokenizer, word_groups, cleaned_text)
            
    except Exception as e:
        print(f"Warning: Using fallback boundary alignment due to: {e}")
        return create_boundaries_word_based(tokens, tokenizer, word_groups, cleaned_text)

def create_boundaries_with_offsets(tokens: List[int], tokenizer, word_groups: List[Tuple[str, bool]], 
                                 cleaned_text: str, offset_mapping: List[Tuple[int, int]]) -> List[bool]:
    """Create boundaries using character offset mapping (most accurate)."""
    
    boundaries = [False] * len(tokens)
    
    word_end_positions = []
    char_pos = 0
    
    for word, is_group_end in word_groups:
        word = word.strip()
        
        word_start = cleaned_text.find(word, char_pos)
        if word_start != -1:
            word_end = word_start + len(word)
            word_end_positions.append((word_end - 1, is_group_end))  
            char_pos = word_end
            
            while char_pos < len(cleaned_text) and cleaned_text[char_pos] == ' ':
                char_pos += 1
        else:
            print(f"Warning: Could not find word '{word}' in cleaned text")
    
    for i, (start, end) in enumerate(offset_mapping):
        if i >= len(boundaries):
            break
            
        if start == 0 and end == 0 and i > 0:
            continue
        
        for word_end_pos, is_group_end in word_end_positions:
            if is_group_end and start <= word_end_pos < end:
                boundaries[i] = True
                break
    
    last_meaningful_idx = len(boundaries) - 1
    while (last_meaningful_idx >= 0 and 
           last_meaningful_idx < len(tokens) and
           tokens[last_meaningful_idx] in [tokenizer.eos_token_id, tokenizer.pad_token_id]):
        last_meaningful_idx -= 1
    
    if last_meaningful_idx >= 0:
        boundaries[last_meaningful_idx] = True
    
    return boundaries

def create_boundaries_word_based(tokens: List[int], tokenizer, word_groups: List[Tuple[str, bool]], 
                                cleaned_text: str) -> List[bool]:
    """Create boundaries using improved word-based alignment (fallback method)."""
    boundaries = [False] * len(tokens)
    
    if not word_groups:
        return boundaries
    
    # Build word-to-group mapping
    words_with_boundaries = []
    for word, is_group_end in word_groups:
        words_with_boundaries.append((word.strip(), is_group_end))
    
    # Try to align with tokens by decoding individual tokens
    try:
        # Decode each token to understand the text structure
        token_texts = []
        for i, token_id in enumerate(tokens):
            if hasattr(tokenizer, 'bos_token_id') and token_id == tokenizer.bos_token_id:
                token_texts.append("[BOS]")
            elif hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
                token_texts.append("[EOS]")
            elif hasattr(tokenizer, 'pad_token_id') and token_id == tokenizer.pad_token_id:
                token_texts.append("[PAD]")
            else:
                try:
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    token_texts.append(token_text)
                except:
                    token_texts.append("[UNK]")
        
        # Reconstruct text and find word boundaries
        reconstructed_text = ""
        token_to_char_map = []
        
        for i, token_text in enumerate(token_texts):
            if token_text.startswith("[") and token_text.endswith("]"):
                # Special token
                token_to_char_map.append((len(reconstructed_text), len(reconstructed_text)))
            else:
                start_pos = len(reconstructed_text)
                reconstructed_text += token_text
                end_pos = len(reconstructed_text)
                token_to_char_map.append((start_pos, end_pos))
        
        # Find where each word ends in the reconstructed text
        char_pos = 0
        for word, is_group_end in words_with_boundaries:
            # Find this word in reconstructed text
            word_start = reconstructed_text.find(word, char_pos)
            if word_start != -1:
                word_end = word_start + len(word) - 1  # Last character of word
                
                if is_group_end:
                     # Find which token contains this character position
                    for i, (token_start, token_end) in enumerate(token_to_char_map):
                        if token_start <= word_end < token_end:
                            boundaries[i] = True
                            break
                
                char_pos = word_start + len(word)
                # Skip spaces
                while char_pos < len(reconstructed_text) and reconstructed_text[char_pos] == ' ':
                    char_pos += 1
        
    except Exception as e:    
        print(f"Warning: Word-based alignment failed ({e}), using simple distribution")
        # Simple fallback: distribute boundaries evenly
        group_count = sum(1 for _, is_end in word_groups if is_end)
        if group_count > 0:
            tokens_per_group = len(tokens) / group_count
            for i in range(group_count):
                boundary_pos = min(int((i + 1) * tokens_per_group) - 1, len(tokens) - 1)
                if boundary_pos >= 0:
                    boundaries[boundary_pos] = True
    
    # Always mark the last meaningful token as boundary
    last_meaningful_idx = len(boundaries) - 1
    while (last_meaningful_idx >= 0 and 
           last_meaningful_idx < len(tokens) and
           hasattr(tokenizer, 'eos_token_id') and hasattr(tokenizer, 'pad_token_id') and
           tokens[last_meaningful_idx] in [tokenizer.eos_token_id, tokenizer.pad_token_id]):
        last_meaningful_idx -= 1
    
    if last_meaningful_idx >= 0:
        boundaries[last_meaningful_idx] = True
    
    return boundaries

def boundaries_for_token_ids(tokenizer, text: str, token_ids: List[int]):

    return word_groups_to_token_boundaries(tokenizer, text, token_ids)
