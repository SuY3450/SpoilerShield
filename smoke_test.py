"""A 코드 확인: 실제 분할에서 리뷰 몇 개를 떼어 와 데이터 → 배치 → 모델이 이어지는지 검사합니다.

  python smoke_test.py               # 작은 임의 모델로 연결만 확인 (빠름)
  python smoke_test.py --pretrained  # 실제 XLM-RoBERTa 가중치로 확인 (약 1GB 다운로드)

원본 데이터는 읽기만 하고, 만드는 파일은 모두 임시 폴더에 두었다가 지웁니다.
"""

import argparse
import json
import tempfile
from itertools import groupby, islice
from pathlib import Path

import torch
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

from model_setup.dataset import create_dataloader, load_tokenized_splits
from model_setup.model import create_model, create_tokenizer
from model_setup.prepare_data import SPLIT_DIR, SPLITS, make_subset, prepare_context_files, read_jsonl


def copy_sample(source_dir, target_dir, num_reviews):
    """각 분할에서 앞쪽 리뷰 num_reviews개를 통째로 복사합니다."""
    target_dir.mkdir(parents=True)
    for split in SPLITS:
        reviews = groupby(read_jsonl(source_dir / f"{split}.jsonl"), key=lambda row: row["review_id"])
        rows = [row for _, group in islice(reviews, num_reviews) for row in group]
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (target_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")


def check_context(original_path, context_rows):
    """모든 행의 prev_text·next_text가 같은 리뷰의 바로 앞·뒤 문장인지 확인합니다."""
    text = {(r["review_id"], r["sentence_index"]): r["text"] for r in read_jsonl(original_path)}
    assert len(context_rows) == len(text)
    for row in context_rows:
        key = row["review_id"], row["sentence_index"]
        assert row["text"] == text[key] and row["label"] in (0, 1)
        assert row["prev_text"] == text.get((key[0], key[1] - 1), "")
        assert row["next_text"] == text.get((key[0], key[1] + 1), "")


def tiny_model(tokenizer):
    config = XLMRobertaConfig(
        vocab_size=len(tokenizer), hidden_size=48, intermediate_size=96, num_hidden_layers=1,
        num_attention_heads=4, num_labels=2, pad_token_id=tokenizer.pad_token_id,
    )
    return XLMRobertaForSequenceClassification(config).eval()


def main():
    parser = argparse.ArgumentParser(description="A 코드 확인")
    parser.add_argument("--pretrained", action="store_true", help="실제 사전학습 모델로 확인")
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--num-reviews", type=int, default=20, help="분할별로 떼어 올 리뷰 수")
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = create_tokenizer()
    model = create_model().eval() if args.pretrained else tiny_model(tokenizer)

    with tempfile.TemporaryDirectory(prefix="spoilershield_a_") as folder:
        folder = Path(folder)
        dirs = dict(split_dir=folder / "splits", context_dir=folder / "context")
        copy_sample(args.split_dir, dirs["split_dir"], args.num_reviews)

        # 1. 문맥 파일: 앞뒤 문장이 같은 리뷰의 바로 앞·뒤 문장인지
        context_files = prepare_context_files(**dirs)
        for split in SPLITS:
            check_context(dirs["split_dir"] / f"{split}.jsonl", list(read_jsonl(context_files[split])))
        print("PASS 문맥 파일: 모든 문장의 앞뒤 문장이 정확함", flush=True)

        # 2. 토큰화·배치·모델: 두 모드 × 두 길이
        for mode in ("baseline", "ours"):
            for max_length in (256, 32):
                data = load_tokenized_splits(tokenizer, mode=mode, max_length=max_length,
                                             data_files=context_files)
                for split in SPLITS:
                    batch = next(iter(create_dataloader(data[split], tokenizer, batch_size=4,
                                                        shuffle=split == "train")))
                    assert set(batch) == {"input_ids", "attention_mask", "labels"}
                    assert batch["input_ids"].shape[1] <= max_length
                    with torch.no_grad():
                        output = model(**batch)
                    assert output.logits.shape == (len(batch["labels"]), 2)
                    assert torch.isfinite(output.loss)
                print(f"PASS 배치→모델: {mode}, max_length={max_length}", flush=True)

        # 3. meta: 문장 정보가 파일 순서대로 정확히 나오는지 (C의 리뷰 단위 평가용)
        data = load_tokenized_splits(tokenizer, mode="ours", data_files=context_files,
                                     splits=("test",), keep_meta=True)
        seen = []
        for batch in create_dataloader(data["test"], tokenizer, batch_size=4):
            meta = batch.pop("meta")
            seen += list(zip(meta["review_id"], meta["sentence_index"], meta["text"], batch["labels"].tolist()))
        expected = [(r["review_id"], r["sentence_index"], r["text"], r["label"])
                    for r in read_jsonl(context_files["test"])]
        assert seen == expected
        print("PASS meta: 리뷰 ID·문장 번호·원문·라벨이 파일 순서와 일치", flush=True)

        # 4. 무작위 표본: 리뷰를 통째로 뽑고, train은 섞되 문맥은 유지되는지
        subset = make_subset("train", num_reviews=5, subset_dir=folder / "subsets", **dirs)
        rows = list(read_jsonl(subset))
        chosen = {row["review_id"] for row in rows}
        originals = [row for row in read_jsonl(context_files["train"]) if row["review_id"] in chosen]
        by_position = lambda row: (row["review_id"], row["sentence_index"])
        assert len(chosen) == 5
        # 1번에서 검증한 문맥 행과 내용이 같으면 섞은 뒤에도 문맥이 그대로라는 뜻입니다.
        assert sorted(rows, key=by_position) == sorted(originals, key=by_position)
        print("PASS 무작위 표본: 리뷰 5개 통째로, 섞은 뒤에도 문맥 유지", flush=True)

    print("완료: A 코드 확인 통과 (분류 성능 평가는 아님)")


if __name__ == "__main__":
    main()
