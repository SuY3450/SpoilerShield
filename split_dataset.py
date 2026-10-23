import json
import hashlib
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("data/processed/sentences.jsonl")
OUTPUT_DIR = Path("data/processed/splits")


def choose_split(review_id):
    """Deterministically assign a whole review to one split."""
    if review_id is None:
        raise ValueError("Missing review_id; cannot safely split by review.")
    digest = hashlib.sha256(str(review_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed file not found: {INPUT_PATH.resolve()}\n"
            "Run preprocess_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: OUTPUT_DIR / f"{name}.jsonl"
        for name in ("train", "validation", "test")
    }

    sentence_counts = Counter()
    label_counts = {name: Counter() for name in ("train", "validation", "test")}
    review_ids = {name: set() for name in ("train", "validation", "test")}
    long_counts = Counter()
    malformed_lines = 0
    missing_review_ids = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as source, \
         open(output_paths["train"], "w", encoding="utf-8") as train_file, \
         open(output_paths["validation"], "w", encoding="utf-8") as val_file, \
         open(output_paths["test"], "w", encoding="utf-8") as test_file:

        targets = {
            "train": train_file,
            "validation": val_file,
            "test": test_file,
        }

        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            review_id = record.get("review_id")
            if review_id is None:
                missing_review_ids += 1
                continue

            split = choose_split(review_id)
            targets[split].write(json.dumps(record, ensure_ascii=False) + "\n")

            sentence_counts[split] += 1
            label_counts[split][record.get("label")] += 1
            review_ids[split].add(review_id)
            if record.get("is_long"):
                long_counts[split] += 1

    total_sentences = sum(sentence_counts.values())
    total_reviews = sum(len(ids) for ids in review_ids.values())

    print("\nData split complete")
    print(f"Input: {INPUT_PATH.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("Split rule: deterministic SHA-256 hash of review_id")
    print("Target proportions: train 80%, validation 10%, test 10%")
    print("Whole reviews stay together; exact ratios may vary slightly.")

    for name in ("train", "validation", "test"):
        count = sentence_counts[name]
        pct = count / total_sentences * 100 if total_sentences else 0
        spoiler_count = label_counts[name][1]
        spoiler_pct = spoiler_count / count * 100 if count else 0
        print(f"\n[{name.upper()}]")
        print(f"Reviews: {len(review_ids[name]):,}")
        print(f"Sentences: {count:,} ({pct:.2f}%)")
        print(f"Non-spoiler (0): {label_counts[name][0]:,}")
        print(f"Spoiler (1): {spoiler_count:,}")
        print(f"Spoiler ratio: {spoiler_pct:.2f}%")
        print(f"Sentences >= 500 characters: {long_counts[name]:,}")
        print(f"Saved to: {output_paths[name]}")

    overlap = (
        review_ids["train"] & review_ids["validation"]
        | review_ids["train"] & review_ids["test"]
        | review_ids["validation"] & review_ids["test"]
    )

    print("\n[VALIDATION]")
    print(f"Total reviews across splits: {total_reviews:,}")
    print(f"Review ID overlap across splits: {len(overlap):,}")
    print(f"Malformed JSON lines skipped: {malformed_lines:,}")
    print(f"Records missing review_id skipped: {missing_review_ids:,}")

    if not overlap and not malformed_lines and not missing_review_ids:
        print("PASS: No review leakage detected; all input records were assigned.")
    else:
        print("CHECK: Review overlap or skipped records need investigation.")


if __name__ == "__main__":
    main()
