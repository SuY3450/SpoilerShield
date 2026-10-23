import json
import gzip
from pathlib import Path

# 원본 파일과 저장 위치
INPUT_PATH = Path("data/raw/goodreads_reviews_spoiler.json")
OUTPUT_PATH = Path("data/processed/sentences.jsonl")

# 지나치게 짧은 문장도 일단 보존한다.
# 긴 문장도 삭제하지 않고 별도로 표시한다.
LONG_SENTENCE_CHARS = 500


def open_input(path):
    """일반 텍스트 파일과 gzip 파일 모두 지원"""
    with open(path, "rb") as f:
        is_gzip = f.read(2) == b"\x1f\x8b"

    if is_gzip:
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_sentence(item):
    """문장 데이터에서 라벨과 텍스트 추출"""
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        label, text = item[0], item[1]
    elif isinstance(item, dict):
        label = item.get("label")
        text = item.get("sentence", item.get("text"))
    else:
        return None

    # 라벨은 0 또는 1이어야 한다.
    if label not in (0, 1, False, True):
        return None

    if not isinstance(text, str):
        return None

    text = text.strip()

    # 빈 문장만 제외
    if not text:
        return None

    return int(label), text


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"원본 데이터 파일을 찾을 수 없습니다: {INPUT_PATH.resolve()}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    review_count = 0
    output_sentence_count = 0
    label_counts = {0: 0, 1: 0}
    skipped_count = 0
    long_sentence_count = 0

    with open_input(INPUT_PATH) as source, open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as target:

        for line in source:
            if not line.strip():
                continue

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                skipped_count += 1
                continue

            review_count += 1

            review_id = review.get("review_id")
            sentences = review.get("review_sentences") or []

            for sentence_index, item in enumerate(sentences):
                parsed = parse_sentence(item)

                if parsed is None:
                    skipped_count += 1
                    continue

                label, text = parsed
                is_long = len(text) >= LONG_SENTENCE_CHARS

                record = {
                    "review_id": review_id,
                    "sentence_index": sentence_index,
                    "text": text,
                    "label": label,
                    "char_length": len(text),
                    "is_long": is_long,
                }

                target.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                output_sentence_count += 1
                label_counts[label] += 1

                if is_long:
                    long_sentence_count += 1

    print("\n전처리 완료")
    print(f"읽은 리뷰 수: {review_count:,}")
    print(f"저장한 문장 수: {output_sentence_count:,}")
    print(f"일반 문장(0): {label_counts[0]:,}")
    print(f"스포일러 문장(1): {label_counts[1]:,}")
    print(f"500자 이상 문장: {long_sentence_count:,}")
    print(f"제외한 레코드 수: {skipped_count:,}")
    print(f"저장 경로: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
