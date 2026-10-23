
import json
from pathlib import Path
from collections import Counter

SPLIT_DIR = Path("data/processed/splits")

for name in ["train", "validation", "test"]:
    path = SPLIT_DIR / f"{name}.jsonl"

    if not path.exists():
        print(f"{name}: 파일을 찾을 수 없습니다.")
        continue

    total = 0
    labels = Counter()
    reviews = set()
    errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                total += 1
                labels[row["label"]] += 1
                reviews.add(row["review_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                errors += 1

    spoiler_ratio = labels[1] / total * 100 if total else 0

    print(f"\n[{name.upper()}]")
    print(f"문장 수: {total:,}")
    print(f"리뷰 수: {len(reviews):,}")
    print(f"일반 문장: {labels[0]:,}")
    print(f"스포일러 문장: {labels[1]:,}")
    print(f"스포일러 비율: {spoiler_ratio:.2f}%")
    print(f"읽기 오류: {errors}")
