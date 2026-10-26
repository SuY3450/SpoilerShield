"""전처리팀 분할 파일로 학습·평가용 파일을 준비합니다. 원본 분할은 수정하지 않습니다.

  문맥 파일 : python -m model_setup.prepare_data --prepare-context
  무작위 표본: python -m model_setup.prepare_data --make-subset --split train --num-reviews <뽑을 리뷰 수>
             (<뽑을 리뷰 수>는 B가 GPU 시간에 맞게 정합니다. 예: 50000)
"""

import argparse
import json
import random
from itertools import groupby
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
CONTEXT_DIR = PROJECT_ROOT / "data" / "processed" / "context"
SUBSET_DIR = PROJECT_ROOT / "data" / "processed" / "subsets"
SPLITS = ("train", "validation", "test")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def iter_context_rows(rows):
    """각 문장에 같은 리뷰의 바로 앞·뒤 문장(prev_text, next_text)을 붙입니다.

    같은 리뷰의 문장은 연속된 행에 있어야 합니다(전처리팀 파일 형식).
    """
    for _, group in groupby(rows, key=lambda row: row["review_id"]):
        sentences = sorted(group, key=lambda row: row["sentence_index"])
        for i, row in enumerate(sentences):
            index = row["sentence_index"]
            before = sentences[i - 1] if i > 0 else None
            after = sentences[i + 1] if i + 1 < len(sentences) else None
            previous = before["text"] if before and before["sentence_index"] == index - 1 else ""
            following = after["text"] if after and after["sentence_index"] == index + 1 else ""
            yield {**row, "prev_text": previous, "next_text": following}


def _write_lines(path, lines):
    """임시 파일에 다 쓴 뒤 교체합니다. 중간에 실패해도 기존 파일은 그대로입니다."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        target.writelines(lines)
    temporary.replace(path)


def _is_fresh(target, source):
    return target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns


def prepare_context_files(splits=SPLITS, force=False, split_dir=SPLIT_DIR, context_dir=CONTEXT_DIR):
    """문맥 파일을 만들고 {split: 경로}를 반환합니다. 이미 최신이면 다시 만들지 않습니다."""
    split_dir, context_dir = Path(split_dir), Path(context_dir)
    context_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in splits:
        source = split_dir / f"{split}.jsonl"
        target = context_dir / f"{split}.jsonl"
        if force or not _is_fresh(target, source):
            print(f"[문맥 생성] {split}", flush=True)
            rows = iter_context_rows(read_jsonl(source))
            _write_lines(target, (json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        paths[split] = str(target)
    return paths


def make_subset(split, num_reviews, seed=42, split_dir=SPLIT_DIR, context_dir=CONTEXT_DIR,
                subset_dir=SUBSET_DIR):
    """리뷰 일부를 무작위로 골라 작은 파일을 만듭니다.

    데이터가 너무 크고(train 약 1,400만 문장) 비슷한 리뷰끼리 모여 있어서,
    앞에서부터 자르지 않고 무작위로 골라 씁니다. train은 순서도 섞습니다.

    num_reviews : 뽑을 리뷰 수. B가 정합니다.
    seed        : 같은 값이면 항상 같은 리뷰가 뽑힙니다.

    각 문장에 앞뒤 문장이 이미 붙어 있어서, 섞어도 문맥은 그대로입니다.
    """
    source = Path(prepare_context_files([split], split_dir=split_dir, context_dir=context_dir)[split])
    target = Path(subset_dir) / f"{split}_{num_reviews}r_seed{seed}.jsonl"
    if _is_fresh(target, source):
        print(f"[건너뜀] 기존 표본 사용: {target.name}", flush=True)
        return str(target)

    rng = random.Random(seed)
    review_ids = list(dict.fromkeys(row["review_id"] for row in read_jsonl(source)))
    chosen = set(rng.sample(review_ids, min(num_reviews, len(review_ids))))
    lines, spoilers = [], 0
    for row in read_jsonl(source):
        if row["review_id"] in chosen:
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
            spoilers += row["label"]
    if split == "train":
        rng.shuffle(lines)

    target.parent.mkdir(parents=True, exist_ok=True)
    _write_lines(target, lines)
    print(f"[표본] {target.name}: 리뷰 {len(chosen):,}개, 문장 {len(lines):,}개, "
          f"스포일러 비율 {spoilers / len(lines):.2%}", flush=True)
    return str(target)


def main():
    parser = argparse.ArgumentParser(description="문맥 파일·무작위 표본 생성")
    parser.add_argument("--prepare-context", action="store_true", help="세 분할의 문맥 파일 생성")
    parser.add_argument("--make-subset", action="store_true", help="리뷰 단위 무작위 표본 생성")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--num-reviews", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prepare_context:
        prepare_context_files()
    if args.make_subset:
        if args.split is None or args.num_reviews is None:
            parser.error("--make-subset에는 --split과 --num-reviews가 필요합니다.")
        make_subset(args.split, args.num_reviews, seed=args.seed)
    if not (args.prepare_context or args.make_subset):
        parser.error("--prepare-context 또는 --make-subset을 지정하세요.")


if __name__ == "__main__":
    main()
