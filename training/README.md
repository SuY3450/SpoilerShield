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
