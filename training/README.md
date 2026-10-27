# Model Training (B)

## 개요

- **담당 역할:** XLM-RoBERTa 학습 파이프라인 구현, Validation 평가, 모델·체크포인트 저장
- **작업 브랜치:** `feat/model-training`
- **작업 범위:** Baseline(현재 문장만 입력) 학습 및 실험. `ours` 입력 모드는 코드에서 지원하지만 아직 학습하지 않음
- **데이터 구분:** Train으로 학습, Validation으로 모델 선택 및 성능 확인. Test는 사용하지 않았으며 최종 평가는 C 담당자와 협의

## 파일별 내용

| 파일 | 내용 |
| --- | --- |
| `training/train.py` | 여러 Epoch 학습, Epoch별 Validation 평가, Best Model·중간 체크포인트 저장 및 재개 |
| `training/README.md` | B 담당 구현 및 실험 기록 |

## 주요 구현

- A 담당자의 `model_setup/model.py`, `model_setup/dataset.py` 사용
- 모델: `FacebookAI/xlm-roberta-base`, 일반(0) / 스포일러(1) 이진 분류
- Optimizer: AdamW, Learning Rate `2e-5`, Batch Size `8`, 최대 입력 길이 `256`, Seed `42`
- Epoch마다 다른 Seed(`42 + epoch`)로 Train 셔플
- Epoch 종료 후 Validation에서 Precision, Recall, F1, TP, FP, FN 확인
- Validation F1이 가장 높은 모델과 토크나이저를 `best_model/`에 저장
- 모델·Optimizer 상태, 학습 위치, 난수 상태를 `latest_checkpoint.pt`에 기록하여 중단 후 재개 지원
- **체크포인트 재개 검증:** 6 Step 연속 학습과 3 Step 학습 후 재개(총 6 Step)를 비교한 결과, 모델 가중치 텐서 201개가 모두 정확히 일치함 (최대 차이 0.0). 단, Epoch 경계 및 장시간 학습에서의 동일성은 추가 검증 필요.

## Baseline 실험 결과

Validation은 동일한 300개 리뷰(3,556문장, 스포일러 39문장)를 사용함.

| Train 표본 | Epoch | Best Validation F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000개 리뷰 (12,137문장) | 1 | 0.0000 | 0 | 0 | 39 |
| 1,000개 리뷰 (12,137문장) | 3 | 0.0455 | 3 | 90 | 36 |
| 10,000개 리뷰 (126,009문장) | 3 | 0.0000 | 0 | 0 | 39 |

1,000개 리뷰를 3 Epoch 학습했을 때 일부 스포일러를 탐지했지만 오탐이 많았음. 10,000개 리뷰를 3 Epoch 학습한 경우 Validation의 모든 문장을 일반 문장으로 예측함. 클래스 불균형(10,000개 표본의 스포일러 3.11%)이 원인 중 하나일 가능성이 있지만 추가 실험이 필요함.

## 학습 실행 예시

프로젝트 루트에서 실행하며, 데이터 경로는 환경에 맞게 변경해야 함.

```bash
python -m training.train \
  --train-file /path/to/train_subset.jsonl \
  --val-file /path/to/validation_subset.jsonl \
  --mode baseline \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5 \
  --max-length 256 \
  --max-train-steps 15752 \
  --max-val-steps 445 \
  --save-steps 2000 \
  --seed 42 \
  --output-dir /path/to/output
```

- `--max-train-steps`는 **Epoch당** 최대 Step 수임. 사용하는 표본 크기에 맞춰 조정해야 함
- `--resume`으로 같은 출력 폴더의 체크포인트에서 재개 시도 가능
- 결과 파일: `best_model/`, `latest_checkpoint.pt`, `history.json`, `config.json`
- 데이터(JSONL), 체크포인트, 모델 가중치는 GitHub에 올리지 않고 별도로 보관

## 향후 논의 사항

- 스포일러 클래스 가중치(Weighted Cross Entropy) 등 불균형 대응 실험
- Recall 향상에 따른 False Positive 증가 여부 평가
- 앞뒤 문맥을 사용하는 `ours` 학습 및 C 담당자의 Test 평가 계획 협의
