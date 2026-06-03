import json
import argparse
from typing import List, Tuple
from transformers import AutoTokenizer


def parse_word_groups(text: str) -> List[Tuple[str, bool]]:
    segments = text.split()
    result = []

    for segment in segments:
        if "##" in segment:
            words_in_segment = segment.split("##")
            for i, word in enumerate(words_in_segment):
                if word.strip():
                    is_group_end = i == len(words_in_segment) - 1
                    result.append((word.strip(), is_group_end))
        else:
            result.append((segment.strip(), True))

    return result


def create_word_boundaries_from_groups(
    tokens: List[int],
    tokenizer,
    word_groups: List[Tuple[str, bool]],
    cleaned_text: str,
) -> List[bool]:
    boundaries = [False] * len(tokens)

    if not word_groups or not tokens:
        return boundaries

    try:
        full_encoding = tokenizer(
            cleaned_text, return_offsets_mapping=True, add_special_tokens=True
        )

        if "offset_mapping" in full_encoding and len(full_encoding["input_ids"]) == len(
            tokens
        ):
            return create_boundaries_with_offsets(
                tokens,
                tokenizer,
                word_groups,
                cleaned_text,
                full_encoding["offset_mapping"],
            )
        return create_boundaries_word_based(
            tokens, tokenizer, word_groups, cleaned_text
        )

    except Exception as e:
        print(f"Warning: Using fallback boundary alignment due to: {e}")
        return create_boundaries_word_based(
            tokens, tokenizer, word_groups, cleaned_text
        )


def create_boundaries_with_offsets(
    tokens: List[int],
    tokenizer,
    word_groups: List[Tuple[str, bool]],
    cleaned_text: str,
    offset_mapping: List[Tuple[int, int]],
) -> List[bool]:
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

            while char_pos < len(cleaned_text) and cleaned_text[char_pos] == " ":
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
    while (
        last_meaningful_idx >= 0
        and last_meaningful_idx < len(tokens)
        and tokens[last_meaningful_idx]
        in [tokenizer.eos_token_id, tokenizer.pad_token_id]
    ):
        last_meaningful_idx -= 1

    if last_meaningful_idx >= 0:
        boundaries[last_meaningful_idx] = True

    return boundaries


def create_boundaries_word_based(
    tokens: List[int],
    tokenizer,
    word_groups: List[Tuple[str, bool]],
    cleaned_text: str,
) -> List[bool]:
    boundaries = [False] * len(tokens)

    if not word_groups:
        return boundaries

    words_with_boundaries = []
    for word, is_group_end in word_groups:
        words_with_boundaries.append((word.strip(), is_group_end))

    try:
        token_texts = []
        for token_id in tokens:
            if hasattr(tokenizer, "bos_token_id") and token_id == tokenizer.bos_token_id:
                token_texts.append("<redacted_BOS>")
            elif hasattr(tokenizer, "eos_token_id") and token_id == tokenizer.eos_token_id:
                token_texts.append("<redacted_EOS>")
            elif hasattr(tokenizer, "pad_token_id") and token_id == tokenizer.pad_token_id:
                token_texts.append("<redacted_PAD>")
            else:
                try:
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    token_texts.append(token_text)
                except Exception:
                    token_texts.append("<redacted_UNK>")

        reconstructed_text = ""
        token_to_char_map = []

        for token_text in token_texts:
            if token_text.startswith("[") and token_text.endswith("]"):
                token_to_char_map.append((len(reconstructed_text), len(reconstructed_text)))
            else:
                start_pos = len(reconstructed_text)
                reconstructed_text += token_text
                end_pos = len(reconstructed_text)
                token_to_char_map.append((start_pos, end_pos))

        char_pos = 0
        for word, is_group_end in words_with_boundaries:
            word_start = reconstructed_text.find(word, char_pos)
            if word_start != -1:
                word_end = word_start + len(word) - 1

                if is_group_end:
                    for ti, (token_start, token_end) in enumerate(token_to_char_map):
                        if token_start <= word_end < token_end:
                            boundaries[ti] = True
                            break

                char_pos = word_start + len(word)
                while char_pos < len(reconstructed_text) and reconstructed_text[char_pos] == " ":
                    char_pos += 1

    except Exception as e:
        print(f"Warning: Word-based alignment failed ({e}), using simple distribution")
        group_count = sum(1 for _, is_end in word_groups if is_end)
        if group_count > 0:
            tokens_per_group = len(tokens) / group_count
            for i in range(group_count):
                boundary_pos = min(int((i + 1) * tokens_per_group) - 1, len(tokens) - 1)
                if boundary_pos >= 0:
                    boundaries[boundary_pos] = True

    last_meaningful_idx = len(boundaries) - 1
    while (
        last_meaningful_idx >= 0
        and last_meaningful_idx < len(tokens)
        and hasattr(tokenizer, "eos_token_id")
        and hasattr(tokenizer, "pad_token_id")
        and tokens[last_meaningful_idx]
        in [tokenizer.eos_token_id, tokenizer.pad_token_id]
    ):
        last_meaningful_idx -= 1

    if last_meaningful_idx >= 0:
        boundaries[last_meaningful_idx] = True

    return boundaries


def process_hindi_data(
    input_file: str,
    output_file: str,
    model_name: str = "/workspace/pranav-shinde/download/Airavata",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                for d in data["hin_Deva"]:
                    original_hindi_text = ""
                    for i in d:
                        original_hindi_text += i + " "

                    word_groups = parse_word_groups(original_hindi_text)

                    cleaned_text = original_hindi_text.replace("##", " ")

                    tokens = tokenizer(cleaned_text, return_tensors=None)["input_ids"]

                    boundaries = create_word_boundaries_from_groups(
                        tokens, tokenizer, word_groups, cleaned_text
                    )

                    processed_entry = {
                        "original_text": original_hindi_text,
                        "cleaned_text": cleaned_text,
                        "tokens": tokens,
                        "word_group_boundaries": boundaries,
                        "word_groups": word_groups,
                    }

                    processed_data.append(processed_entry)

                if (line_num + 1) % 1000 == 0:
                    print(f"Processed {line_num + 1} lines...")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(processed_data)} entries and saved to {output_file}")
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="/nfs/kundeshwar/pranav-shinde/download/Airavata",
    )

    args = parser.parse_args()

    process_hindi_data(args.input, args.output, args.model)