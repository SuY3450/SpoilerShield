# SpoilerShield Baseline Training

## 공통 학습 설정

10K와 20K는 동일한 train.py와 설정으로 학습합니다.
Train 파일과 결과 저장 경로만 다르게 지정합니다.

- Model: FacebookAI/xlm-roberta-base
- Mode: baseline
- Optimizer: AdamW
- Epochs: 3
- Batch Size: 8
- Learning Rate: 2e-5
- Max Length: 256
- Seed: 42
- Checkpoint Interval: 1000 steps
- Validation: 원본 validation.jsonl 전체 (1,830,800문장)
- Best Model: 매 Epoch 전체 Validation F1 기준

## 실행 방법

프로젝트 루트에서 실행합니다.

```bash
python -u -m training.train --train-file TRAIN_PATH --val-file VAL_PATH --output-dir OUTPUT_PATH --mode baseline --epochs 3 --batch-size 8 --lr 2e-5 --max-length 256 --seed 42 --save-steps 1000 --expected-val-samples 1830800 --expected-val-spoilers 56706
```

- TRAIN_PATH: 각자 담당하는 10K 또는 20K Train JSONL 경로
- VAL_PATH: 동일한 원본 processed/splits/validation.jsonl 경로
- OUTPUT_PATH: 각자 별도의 새 결과 폴더
- 학습 재개 시 --resume 추가
- Test 데이터는 Best Model 선정에 사용하지 않음

## 결과 파일

- best_model/: 전체 Validation F1 기준 Best Model
- latest_checkpoint.pt: 학습 재개용 체크포인트
- history.json: Epoch별 학습 및 Validation 결과
- config.json: 실행 설정

데이터셋과 모델 가중치는 GitHub에 업로드하지 않습니다.

## 10K Baseline 실험 결과

### 데이터 및 학습 환경

- Train: 리뷰 10,000개 / 문장 126,009개
- Train Spoiler: 3,921문장 (약 3.11%)
- Validation: 원본 전체 1,830,800문장
- Validation Spoiler: 56,706문장 (약 3.10%)
- 학습 환경: Google Colab Pro / NVIDIA Tesla T4
- 총 학습 Step: 47,256
- 학습 상태: 정상 완료 (종료 코드 0)

### Epoch별 Validation 결과

| 지표 | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Train Loss | 0.141016 | 0.139802 | 0.139483 |
| Precision | 0.0000 | 0.0000 | 0.0000 |
| Recall | 0.0000 | 0.0000 | 0.0000 |
| F1 Score | 0.0000 | 0.0000 | 0.0000 |
| TP | 0 | 0 | 0 |
| FP | 0 | 0 | 0 |
| TN | 1,774,094 | 1,774,094 | 1,774,094 |
| FN | 56,706 | 56,706 | 56,706 |

- **Best Validation F1: 0.0000**
- **Best Epoch: 1** (세 Epoch의 F1이 모두 0으로 동률)
- **Test 평가: 미실시**

### 결과 분석

3개 Epoch 모두 전체 Validation 데이터 1,830,800문장을 평가했으며,
실제 Spoiler 문장 수 56,706개를 검증했다.

모든 Epoch에서 모델이 전체 문장을 Non-spoiler로 예측하여
Precision, Recall, F1 Score가 모두 0으로 나타났다.

학습 데이터의 클래스 불균형(스포일러 약 3.11%)이
영향을 미쳤을 가능성이 있으나, 원인은 아직 확정되지 않았다.
추후 라벨 매핑 및 예측 확률 분포를 확인할 예정이다.

### 결과 저장 위치

Google Drive의 `SpoilerShield_Baseline_v2/` 기준:

- 모델 및 체크포인트: `baseline_10000r_retry/`
- Best Model: `baseline_10000r_retry/best_model/`
- 최종 체크포인트: `baseline_10000r_retry/latest_checkpoint.pt`
- Epoch별 결과: `baseline_10000r_retry/history.json`
- 실행 설정: `baseline_10000r_retry/config.json`
- 전체 학습 로그: `10k_retry.log`

20K Baseline 결과는 별도 실험 완료 후 추가한다.

