"""XLM-RoBERTa 토크나이저와 이진 분류 모델(0 일반, 1 스포일러)을 불러옵니다."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "FacebookAI/xlm-roberta-base"


def create_tokenizer(model_name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(str(model_name), use_fast=True)


def create_model(model_name=MODEL_NAME):
    """기본은 사전학습 모델 + 새 분류층. 체크포인트 폴더 경로를 넣으면 학습된 모델을 불러옵니다."""
    return AutoModelForSequenceClassification.from_pretrained(
        str(model_name),
        num_labels=2,
        id2label={0: "NON_SPOILER", 1: "SPOILER"},
        label2id={"NON_SPOILER": 0, "SPOILER": 1},
    )
