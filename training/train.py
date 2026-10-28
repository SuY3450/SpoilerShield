"""SpoilerShield baseline training with full-validation model selection.

Train and validate on existing, disjoint processed splits. Every epoch evaluates
ALL validation rows; the best model is selected by full-validation F1.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from model_setup.model import create_model, create_tokenizer
from model_setup.dataset import load_tokenized_splits, create_dataloader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, model, optimizer, state, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "settings": vars(args),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    print(f"[Checkpoint] next_epoch={state['epoch'] + 1}, step={state['step']}", flush=True)


def load_checkpoint(path, model, optimizer, device, args):
    # Only load checkpoints created by you: they contain pickled Python RNG states.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("train_file", "val_file", "mode", "batch_size", "lr", "max_length", "seed", "expected_val_samples", "expected_val_spoilers"):
        if payload["settings"][key] != getattr(args, key):
            raise ValueError(f"Checkpoint setting mismatch: {key}")
    if args.epochs < payload["state"]["epoch"]:
        raise ValueError("Requested --epochs is smaller than completed epochs")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.state.values():
        for key, value in group.items():
            if isinstance(value, torch.Tensor):
                group[key] = value.to(device)
    return payload


def restore_rng(payload):
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    if torch.cuda.is_available() and "cuda_rng" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def load_data(tokenizer, args):
    return load_tokenized_splits(
        tokenizer,
        mode=args.mode,
        max_length=args.max_length,
        data_files={"train": args.train_file, "validation": args.val_file},
        splits=("train", "validation"),
        streaming=True,
    )


@torch.inference_mode()
def validate_full(model, loader, device, expected_samples, expected_spoilers):
    model.eval()
    tp = fp = tn = fn = 0
    processed = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        predictions = logits.argmax(dim=-1)
        labels = batch["labels"]
        tp += int(((predictions == 1) & (labels == 1)).sum().item())
        fp += int(((predictions == 1) & (labels == 0)).sum().item())
        tn += int(((predictions == 0) & (labels == 0)).sum().item())
        fn += int(((predictions == 0) & (labels == 1)).sum().item())
        count = labels.numel()
        processed += count
        if processed % 100000 < count:
            print(f"  Validation: {processed:,}/{expected_samples:,}", flush=True)
        if processed > expected_samples:
            raise ValueError(f"Validation exceeds expected count: {processed} > {expected_samples}")
    if processed != expected_samples:
        raise ValueError(f"INCOMPLETE validation: {processed} != {expected_samples}. No best model selected.")
    if tp + fp + tn + fn != processed:
        raise AssertionError("Confusion matrix counts do not match")
    actual_spoilers = tp + fn
    if actual_spoilers != expected_spoilers:
        raise ValueError(
            f"Validation label count mismatch: Spoiler={actual_spoilers:,}, "
            f"expected={expected_spoilers:,}. No best model selected."
        )
    print(f"  Validation verified: {processed:,} rows, {actual_spoilers:,} spoilers", flush=True)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(val_samples=processed, tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1, fpr=fpr,
                actual_spoilers=tp + fn, predicted_spoilers=tp + fp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["baseline", "ours"], default="baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--expected-val-samples", type=int, required=True,
                        help="Exact full validation sentence count; fail closed if different")
    parser.add_argument("--expected-val-spoilers", type=int, default=56706,
                        help="Exact spoiler count in the original full validation split (default: 56706)")
    parser.add_argument("--save-all-epochs", action="store_true",
                        help="Also keep each epoch's complete model/tokenizer")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.save_steps < 1 or args.expected_val_samples < 1:
        parser.error("epochs, batch size, save steps and expected val samples must be positive")
    if not 0 <= args.expected_val_spoilers <= args.expected_val_samples:
        parser.error("expected-val-spoilers must be between 0 and expected-val-samples")
    for file in (args.train_file, args.val_file):
        if not Path(file).is_file():
            parser.error(f"File does not exist: {file}")
    if Path(args.train_file).resolve() == Path(args.val_file).resolve():
        parser.error("Train and validation paths must differ")

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "latest_checkpoint.pt"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        parser.error("Output dir is not empty; choose a NEW directory or pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    tokenizer = create_tokenizer()
    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    state = {"epoch": 0, "step": 0, "global_step": 0, "loss_sum": 0.0,
             "loss_count": 0, "best_f1": -1.0, "best_epoch": None, "history": []}
    resume_payload = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        resume_payload = load_checkpoint(checkpoint_path, model, optimizer, device, args)
        state = resume_payload["state"]
        print(f"Resuming epoch {state['epoch'] + 1}, batch {state['step']}", flush=True)

    data = load_data(tokenizer, args)
    write_json_atomic(output_dir / "config.json", vars(args))

    for epoch in range(state["epoch"], args.epochs):
        # Keep RNG replay compatible with the current dataset loader implementation.
        train_loader = create_dataloader(data["train"], tokenizer,
                                         batch_size=args.batch_size, shuffle=True,
                                         seed=args.seed + epoch)
        iterator = iter(train_loader)
        skip = state["step"] if epoch == state["epoch"] else 0
        for _ in range(skip):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError("Checkpoint step exceeds train dataset length") from exc
        if resume_payload is not None:
            # Data-loader creation and skipped batches can consume RNG. Restore the
            # saved training RNG only after reaching the resume boundary.
            restore_rng(resume_payload)
            resume_payload = None
        model.train()
        processed_batches = skip
        print(f"Epoch {epoch + 1}/{args.epochs}, skip={skip}", flush=True)
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            processed_batches += 1
            state["step"] = processed_batches
            state["global_step"] += 1
            state["loss_sum"] += float(loss.item())
            state["loss_count"] += 1
            if state["global_step"] % 100 == 0:
                print(f"  step={state['global_step']}, train_loss={loss.item():.5f}", flush=True)
            if state["global_step"] % args.save_steps == 0:
                save_checkpoint(checkpoint_path, model, optimizer, state, args)
        if processed_batches == 0:
            raise ValueError("Empty training dataset")

        avg_train_loss = state["loss_sum"] / max(1, state["loss_count"])
        val_loader = create_dataloader(data["validation"], tokenizer,
                                       batch_size=args.batch_size, shuffle=False)
        print(f"Epoch {epoch + 1}: full validation starting", flush=True)
        metrics = validate_full(model, val_loader, device, args.expected_val_samples, args.expected_val_spoilers)
        metrics.update(epoch=epoch + 1, train_loss=avg_train_loss, train_batches=processed_batches)
        if args.save_all_epochs:
            epoch_dir = output_dir / f"epoch_{epoch + 1}"
            model.save_pretrained(epoch_dir, safe_serialization=True)
            tokenizer.save_pretrained(epoch_dir)
        if metrics["f1"] > state["best_f1"]:
            best_dir = output_dir / "best_model"
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            state["best_f1"] = metrics["f1"]
            state["best_epoch"] = epoch + 1
            print(f"  NEW BEST: epoch {epoch + 1}, F1={metrics['f1']:.6f}", flush=True)
        state["history"].append(metrics)
        write_json_atomic(output_dir / "history.json", state["history"])
        state["epoch"] = epoch + 1
        state["step"] = 0
        state["loss_sum"] = 0.0
        state["loss_count"] = 0
        save_checkpoint(checkpoint_path, model, optimizer, state, args)
        print(f"Epoch {epoch + 1} complete: {metrics}", flush=True)

    print(f"Done. Best epoch: {state['best_epoch']}, F1: {state['best_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
