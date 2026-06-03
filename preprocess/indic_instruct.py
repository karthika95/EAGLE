import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    here = Path(__file__).resolve().parent
    default_out = here / "indic_instruct_align.jsonl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(default_out))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.output)

    dataset = load_dataset(
        "ai4bharat/indic-align", "Wiki_Chat", split="train", streaming=True
    )

    # languages = ["ben_Beng", "hin_Deva", "tam_Taml", "tel_Telu"]
    languages = ["hin_Deva", ]

    first_row = next(iter(dataset))
    all_keys = first_row.keys()
    lang_to_remove = [c for c in all_keys if c not in languages]

    new_dataset = dataset.remove_columns(lang_to_remove)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in new_dataset:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"wrote {n} lines -> {out_path}")


if __name__ == "__main__":
    main()