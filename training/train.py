"""SpoilerShield: epoch shuffle + checkpoint resume."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from model_setup.model import create_model, create_tokenizer
from model_setup.dataset import load_tokenized_splits, create_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, state, args):
    saved = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state.copy(),
        "settings": vars(args),
        "rng_torch": torch.get_rng_state(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
    }

    if torch.cuda.is_available():
        saved["rng_cuda"] = torch.cuda.get_rng_state_all()

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(saved, temp)
    temp.replace(path)

    print(
        f"[Checkpoint] Epoch {state['epoch'] + 1}, "
        f"Step {state['step']} 저장 완료",
        flush=True,
    )


def load_checkpoint(path, model, optimizer, args, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)

    keys = [
        "train_file", "val_file", "mode", "epochs",
        "batch_size", "lr", "max_length", "seed",
        "max_train_steps", "max_val_steps",
    ]

    for key in keys:
        if saved["settings"][key] != getattr(args, key):
            raise ValueError(f"설정 불일치: {key}")

    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])

    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.to(device)

    return saved


def restore_rng(saved):
    torch.set_rng_state(saved["rng_torch"])
    random.setstate(saved["rng_python"])
    np.random.set_state(saved["rng_numpy"])

    if torch.cuda.is_available() and "rng_cuda" in saved:
        torch.cuda.set_rng_state_all(saved["rng_cuda"])


@torch.no_grad()
def validate(model, loader, device, max_steps):
    model.eval()
    losses, predictions, labels = [], [], []

    for i, batch in enumerate(loader):
        if i >= max_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)

        losses.append(output.loss.item())
        predictions.extend(output.logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())

    if not labels:
        raise ValueError("Validation 데이터가 비어 있습니다.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary",
        pos_label=1, zero_division=0
    )

    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))

    return {
        "val_loss": float(np.mean(losses)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "val_samples": len(labels),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["baseline", "ours"], default="baseline")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-train-steps", type=int, default=15752)
    p.add_argument("--max-val-steps", type=int, default=445)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")

    # 재개 기능 테스트용: 지정한 전체 Step에서 정상 중단
    p.add_argument("--stop-after-steps", type=int, default=None)

    args = p.parse_args()

    if min(args.epochs, args.batch_size, args.max_train_steps,
           args.max_val_steps, args.save_steps) < 1:
        p.error("Epoch 및 Step 설정은 1 이상이어야 합니다.")

    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        p.error("--stop-after-steps는 1 이상이어야 합니다.")

    for filename in [args.train_file, args.val_file]:
        if not Path(filename).is_file():
            p.error(f"파일 없음: {filename}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest_checkpoint.pt"

    tokenizer = create_tokenizer()
    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data = load_tokenized_splits(
        tokenizer,
        mode=args.mode,
        max_length=args.max_length,
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
        splits=("train", "validation"),
        streaming=True,
    )

    state = {
        "epoch": 0,
        "step": 0,
        "global_step": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
        "best_f1": -1.0,
        "history": [],
    }

    saved = None
    if args.resume:
        saved = load_checkpoint(
            checkpoint, model, optimizer, args, device
        )
        state = saved["state"]

        print(
            f"재개 위치: Epoch {state['epoch'] + 1}, "
            f"Step {state['step']}",
            flush=True,
        )

    print("Device:", device, flush=True)

    for epoch in range(state["epoch"], args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====", flush=True)

        train_loader = create_dataloader(
            data["train"], tokenizer,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        )

        iterator = iter(train_loader)
        skip = state["step"] if epoch == state["epoch"] else 0

        for _ in range(skip):
            try:
                next(iterator)
            except StopIteration:
                raise RuntimeError("재개 위치가 데이터 범위를 초과했습니다.")

        # DataLoader 재구성과 건너뛰기가 끝난 후 난수 상태 복원
        if saved is not None and epoch == state["epoch"]:
            restore_rng(saved)

        model.train()

        while state["step"] < args.max_train_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()

            state["step"] += 1
            state["global_step"] += 1
            state["loss_sum"] += loss.item()
            state["loss_count"] += 1

            if state["step"] % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | Step {state['step']} "
                    f"| Loss {loss.item():.4f}",
                    flush=True,
                )

            if state["global_step"] % args.save_steps == 0:
                save_checkpoint(
                    checkpoint, model, optimizer, state, args
                )

            if (
                args.stop_after_steps is not None
                and state["global_step"] >= args.stop_after_steps
            ):
                save_checkpoint(
                    checkpoint, model, optimizer, state, args
                )
                print("테스트용 중단 완료", flush=True)
                return

        val_loader = create_dataloader(
            data["validation"], tokenizer,
            batch_size=args.batch_size,
            shuffle=False,
        )

        metrics = validate(
            model, val_loader, device, args.max_val_steps
        )

        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = (
            state["loss_sum"] / state["loss_count"]
        )
        metrics["global_step"] = state["global_step"]

        state["history"].append(metrics)

        print("Validation:", metrics, flush=True)

        if metrics["f1"] > state["best_f1"]:
            state["best_f1"] = metrics["f1"]
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            print("Best Model 저장 완료!", flush=True)

        (output_dir / "history.json").write_text(
            json.dumps(state["history"], indent=2),
            encoding="utf-8",
        )
        (output_dir / "config.json").write_text(
            json.dumps(vars(args), indent=2),
            encoding="utf-8",
        )

        # 다음 Epoch 시작 위치로 변경
        state["epoch"] = epoch + 1
        state["step"] = 0
        state["loss_sum"] = 0.0
        state["loss_count"] = 0

        save_checkpoint(
            checkpoint, model, optimizer, state, args
        )

        saved = None

    print("\n학습 완료!", flush=True)
    print("Best Validation F1:", round(state["best_f1"], 4))


if __name__ == "__main__":
    main()
