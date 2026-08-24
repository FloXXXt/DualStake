"""Build the paper's NQ + HotpotQA training data from Search-R1-compatible JSONL.

Expected JSONL fields: ``question`` and ``golden_answers``. The input directory must
contain ``nq/{train,test}.jsonl`` and ``hotpotqa/{train,dev}.jsonl``.
"""

import argparse
from pathlib import Path

import datasets as hf_datasets

SPLITS = {
    "nq": ("nq", "train.jsonl", "test.jsonl"),
    "hotpotqa": ("hotpotqa", "train.jsonl", "dev.jsonl"),
}


def make_prompt(question: str) -> str:
    return (
        "Answer the given question. You can search for external knowledge using "
        "<search> query </search>. Search results will be returned between "
        "<information> and </information>. You can search as many times as you want. "
        "After receiving search results, first output your current confidence in "
        "<confidence> X </confidence> (integer 1-10), then either search again or "
        "give your answer. When giving your final answer, write "
        "<answer> your answer </answer> followed by <final-confidence> X "
        "</final-confidence> (integer 1-10). "
        f"Question: {question}"
    )


def load_and_convert(source: Path, dataset_name: str, split: str):
    dataset = hf_datasets.load_dataset("json", data_files=str(source), split="train")

    def convert(example, index):
        question = example["question"].strip()
        if not question.endswith("?"):
            question += "?"
        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": make_prompt(question)}],
            "ability": "fact-reasoning",
            "reward_model": {"style": "rule", "ground_truth": {"target": example["golden_answers"]}},
            "extra_info": {"split": split, "index": index, "dataset": dataset_name},
        }

    return dataset.map(convert, with_indices=True, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing NQ and HotpotQA JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/nq_hotpotqa"))
    args = parser.parse_args()

    converted = {"train": [], "test": []}
    for dataset_name, (subdir, train_name, test_name) in SPLITS.items():
        for split, filename in (("train", train_name), ("test", test_name)):
            source = args.raw_root / subdir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing input file: {source}")
            converted[split].append(load_and_convert(source, dataset_name, split))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, parts in converted.items():
        combined = hf_datasets.concatenate_datasets(parts)
        combined.to_parquet(str(args.output_dir / f"{split}.parquet"))


if __name__ == "__main__":
    main()
