"""max_length에서 문장이 잘리는지 확인합니다. encode_row와 같은 규칙으로 길이를 셉니다.

  python check_length.py                         # train·validation·test 전체, max_length 256
  python check_length.py --limit 20000           # 분할마다 앞 2만 문장만 (빠른 확인)
  python check_length.py --splits validation     # 한 분할만
  python check_length.py --max-length 512        # 다른 길이로 확인
"""

import argparse
from itertools import islice

from model_setup.dataset import _neighbour_budgets
from model_setup.model import create_tokenizer
from model_setup.prepare_data import SPLIT_DIR, SPLITS, iter_context_rows, read_jsonl


def batches(rows, size=20_000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def check_split(split, tokenizer, max_length, limit=0):
    """한 분할을 확인하고 (문장 수, baseline 잘림, ours 문장 잘림, ours 문맥 잘림, 잘린 문장 예시)를 반환."""
    budget = max_length - 4  # ours: <s> 문장 </s></s> 문맥 </s>
    n = baseline_cut = target_cut = context_cut = longest = 0
    examples = []
    # 원본 분할에서 바로 앞뒤 문장을 붙입니다 (문맥 파일과 같은 함수, 파일은 만들지 않음).
    rows = iter_context_rows(read_jsonl(SPLIT_DIR / f"{split}.jsonl"))
    if limit:
        rows = islice(rows, limit)
    for batch in batches(rows):
        encode = lambda texts: tokenizer(texts, add_special_tokens=False)["input_ids"]
        targets = encode([row["text"] for row in batch])
        prevs = encode(["Previous: " + row["prev_text"] for row in batch])
        nexts = encode(["Next: " + row["next_text"] for row in batch])
        for row, target, prev, nxt in zip(batch, targets, prevs, nexts):
            n += 1
            longest = max(longest, len(target))
            baseline_cut += len(target) > max_length - 2  # baseline: <s> 문장 </s>
            if len(target) > budget:
                target_cut += 1
                examples.append((len(target), row["review_id"], row["sentence_index"], row["text"]))
                continue
            prev_len = len(prev) if row["prev_text"] else 0
            next_len = len(nxt) if row["next_text"] else 0
            kept_prev, kept_next = _neighbour_budgets(prev_len, next_len, budget - len(target))
            context_cut += kept_prev < prev_len or kept_next < next_len
        print(f"  [{split}] {n:,}개 확인", flush=True)
    return n, baseline_cut, target_cut, context_cut, longest, examples


def main():
    parser = argparse.ArgumentParser(description="max_length 잘림 확인")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="분할마다 확인할 문장 수 (0이면 전체)")
    args = parser.parse_args()

    tokenizer = create_tokenizer()
    results = {split: check_split(split, tokenizer, args.max_length, args.limit) for split in args.splits}

    print(f"\n===== max_length {args.max_length} 결과 =====")
    for split, (n, baseline_cut, target_cut, context_cut, longest, examples) in results.items():
        share = lambda count: f"{count:,}개 ({count / n:.4%})"
        print(f"[{split}] 문장 {n:,}개, 가장 긴 문장 {longest} 토큰")
        print(f"  baseline: 판정할 문장이 잘림 {share(baseline_cut)}")
        print(f"  ours    : 판정할 문장이 잘림 {share(target_cut)}")
        print(f"  ours    : 앞뒤 문장 일부가 잘림 {share(context_cut)}")
        for length, review_id, index, text in sorted(examples, reverse=True)[:5]:
            print(f"    잘린 문장 예) {length} 토큰, review {review_id} #{index}: {text[:80]}...")


if __name__ == "__main__":
    main()
