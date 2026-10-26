"""문장을 모델 입력으로 토큰화하고 DataLoader를 만듭니다.

배치 형식
  input_ids, attention_mask : 모델 입력 [B, L]
  labels                    : 현재 문장의 정답 [B] (0 일반, 1 스포일러)
  meta                      : keep_meta=True일 때만. 문장 정보 리스트
                              {"review_id", "book_id", "sentence_index", "text"}
                              모델에 넣기 전에 batch.pop("meta")로 꺼냅니다.
"""

from .prepare_data import SPLITS, prepare_context_files

MODES = ("baseline", "ours")  # baseline: 현재 문장만, ours: 현재 문장 + 앞뒤 문장
META_COLUMNS = ("review_id", "book_id", "sentence_index", "text")


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode는 {MODES} 중 하나여야 합니다: {mode!r}")


def _neighbour_budgets(prev_len, next_len, remaining):
    """남은 자리를 앞뒤 문맥에 반씩 나누고, 한쪽이 짧으면 남는 자리를 다른 쪽에 줍니다."""
    n_prev = min(prev_len, remaining // 2)
    n_next = min(next_len, remaining - n_prev)
    n_prev = min(prev_len, remaining - n_next)
    return n_prev, n_next


def encode_row(row, tokenizer, mode="ours", max_length=256):
    """현재 문장을 판정 대상으로 토큰화합니다. 정답은 항상 현재 문장의 label입니다.

    baseline: <s> 현재 문장 </s>
    ours    : <s> 현재 문장 </s></s> Previous: 앞 문장 Next: 뒤 문장 </s>
              현재 문장을 먼저 넣고, 남는 자리에 앞 문장의 끝과 뒤 문장의 시작을 넣습니다.
    """
    _check_mode(mode)
    if mode == "baseline":
        encoded = tokenizer(row["text"], max_length=max_length, truncation=True)
    else:
        budget = max_length - tokenizer.num_special_tokens_to_add(pair=True)
        current = tokenizer.encode(row["text"], add_special_tokens=False)[:budget]
        previous, following = row.get("prev_text") or "", row.get("next_text") or ""
        prev_ids = tokenizer.encode("Previous: " + previous, add_special_tokens=False) if previous else []
        next_ids = tokenizer.encode("Next: " + following, add_special_tokens=False) if following else []
        n_prev, n_next = _neighbour_budgets(len(prev_ids), len(next_ids), budget - len(current))
        context = (prev_ids[-n_prev:] if n_prev else []) + next_ids[:n_next]
        ids = tokenizer.build_inputs_with_special_tokens(current, context)
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    if "label" in row:
        encoded["labels"] = int(row["label"])
    return encoded


def load_tokenized_splits(tokenizer, mode="ours", max_length=256, data_files=None,
                          splits=SPLITS, keep_meta=False, streaming=True):
    """분할별 토큰화 데이터셋을 {split: dataset}으로 반환합니다.

    data_files : {"train": 경로, ...}. make_subset() 결과 등을 지정합니다.
                 없는 분할은 문맥 파일 전체를 씁니다(baseline·ours 공통, baseline은 문맥을 쓰지 않음).
    splits     : 불러올 분할. 예) 학습 ("train", "validation"), 평가 ("test",)
    keep_meta  : True면 배치에 "meta"(문장 정보)를 함께 넘깁니다.
    """
    from datasets import load_dataset

    _check_mode(mode)
    files = {split: str(path) for split, path in (data_files or {}).items() if split in splits}
    missing = [split for split in splits if split not in files]
    if missing:
        files.update(prepare_context_files(missing))
    raw = load_dataset("json", data_files=files, streaming=streaming)
    keep = META_COLUMNS if keep_meta else ()
    result = {}
    for split in splits:
        columns = list(next(iter(raw[split])))
        if mode == "ours" and "prev_text" not in columns:
            raise ValueError(f"{split}: ours 모드에는 문맥 파일이 필요합니다. 받은 파일: {files[split]}")
        result[split] = raw[split].map(
            lambda row: encode_row(row, tokenizer, mode=mode, max_length=max_length),
            remove_columns=[column for column in columns if column not in keep],
        )
    return result


class SpoilerCollator:
    """모델 입력은 패딩해서 텐서로 만들고, 문장 정보는 batch["meta"]에 리스트로 담습니다."""

    def __init__(self, tokenizer):
        from transformers import DataCollatorWithPadding

        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features):
        meta = {column: [f[column] for f in features] for column in META_COLUMNS if column in features[0]}
        batch = self.pad([{k: v for k, v in f.items() if k not in META_COLUMNS} for f in features])
        if meta:
            batch["meta"] = meta
        return batch


def create_dataloader(dataset, tokenizer, batch_size, shuffle=False, seed=42, buffer_size=10_000):
    """문장을 batch_size개씩 묶어 넘겨주는 DataLoader를 만듭니다.

    batch_size : 한 번에 묶을 문장 수. 학습 설정이므로 기본값이 없고 B·C가 직접 정합니다.
                 예) 학습 create_dataloader(data["train"], tokenizer, batch_size=16, shuffle=True)
    shuffle    : train만 True. 스트리밍 데이터는 buffer_size 안에서만 섞이므로
                 train은 make_subset()으로 미리 섞은 파일을 쓰세요.
    """
    import torch
    from datasets import IterableDataset
    from torch.utils.data import DataLoader

    streaming = isinstance(dataset, IterableDataset)
    if streaming and shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not streaming,
        collate_fn=SpoilerCollator(tokenizer),
        generator=torch.Generator().manual_seed(seed),
    )
