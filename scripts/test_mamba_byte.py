# %%
from itertools import chain
import os
import torch
from datasets import load_dataset
from pmnet import ByteTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

try:
    from mamba_ssm.utils.generation import InferenceParams
except Exception:  # 환경/버전별 fallback
    InferenceParams = None

# [CHANGE] 체크포인트/ID 교체
CKPT_PATH = "JunxiongWang/MambaByte_PG19_353M"

DATA_PROCESS_BATCH_SIZE = 1000
NUM_PROC = 16
SEQ_LENGTH = 1024 * 1024
DATASET_ID = "emozilla/pg19"
NUM_EVAL_SAMPLES = 4
loss_save_dir = "../data/mambabyte_losses_1m"
DATA_SPLIT = "test"
os.makedirs(loss_save_dir, exist_ok=True)

import torch


tokenizer = ByteTokenizer()
test_dataset = load_dataset(DATASET_ID, split=DATA_SPLIT)
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_batch(examples):
    tokenized = tokenizer(examples["text"])
    input_ids = tokenized["input_ids"]

    input_ids = [ids[:SEQ_LENGTH] for ids in input_ids if len(ids) >= SEQ_LENGTH]
    result = {"input_ids": input_ids}
    return result


test_dataset = test_dataset.map(
    process_batch,
    batched=True,
    batch_size=DATA_PROCESS_BATCH_SIZE,
    remove_columns=test_dataset.column_names,
    num_proc=NUM_PROC,
    desc="Tokenizing",
)

print(f"Number of test samples: {len(test_dataset)}")
# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _extract_logits(outputs):
    # MambaLMHeadModel은 보통 logits를 반환하거나 (logits, ...) 튜플을 반환함
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    return outputs


def get_loss_over_positions(model, input_ids, ablation=False, chunk_size=1024 * 30):
    # [CHANGE] MambaByte inference_params는 현재 1-token decoding만 지원하는 경로로 들어가므로
    # 긴 시퀀스 loss 계산에서는 사용하지 않음.
    if ablation:
        raise NotImplementedError(
            "MambaByte ablation is not implemented in this script."
        )

    model.eval()
    seq_len = len(input_ids)
    all_losses = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_input = input_ids[i:end_idx]

            label_end_idx = min(i + chunk_size + 1, seq_len)
            chunk_labels_ids = input_ids[i + 1 : label_end_idx]

            input_tensor = torch.tensor(
                chunk_input,
                dtype=torch.long,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ).unsqueeze(0)

            # [CHANGE] inference_params=None 강제 (에러 방지)
            outputs = model(input_ids=input_tensor)
            logits = _extract_logits(outputs)

            if len(chunk_labels_ids) == len(chunk_input):
                shift_logits = logits.contiguous()
                shift_labels = torch.tensor(
                    chunk_labels_ids,
                    dtype=torch.long,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_tensor[..., 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            all_losses.extend(loss.detach().float().cpu().numpy().tolist())

            del outputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_losses


def evaluate_longbench_sample():
    # [CHANGE] MambaByte 로드
    model = MambaLMHeadModel.from_pretrained(
        CKPT_PATH,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    param_size = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {param_size:,}")
    print(f"규모 약어: {param_size / 1_000_000_000:.3f}B")

    dataset_for_eval = test_dataset.select(range(NUM_EVAL_SAMPLES))
    losses_list = []

    for i, sample in enumerate(
        tqdm(dataset_for_eval, desc="Evaluating Samples", total=len(dataset_for_eval))
    ):
        save_path = f"{loss_save_dir}/sample_{i}_losses.npy"
        if os.path.exists(save_path):
            losses = np.load(save_path)
            losses_list.append(losses)
            continue

        losses = get_loss_over_positions(model, sample["input_ids"], ablation=False)
        np.save(save_path, np.array(losses))
        losses_list.append(losses)

    losses_mean = np.array(losses_list).mean(axis=0)

    plt.figure(figsize=(12, 6))

    def moving_average(a, n=1000):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    smooth = moving_average(losses_mean) / 0.693  # BPB 보정
    x = range(len(smooth))

    plt.plot(
        x,
        smooth,
        label="MambaByte (PG-19)",
        color="blue",
        alpha=0.6,
    )

    plt.title(f"Bits Per Byte (MambaByte)\nSeq Len: {SEQ_LENGTH//1024}k Tokens")
    plt.xlabel("Token Position")
    plt.ylabel("BPB (Bits Per Byte)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = "mambabyte_bpb.png"
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")

    print(f"Avg Loss: {np.mean(losses_mean):.4f}")
    print(f"PPL: {np.exp(np.mean(losses_mean)):.2f}")


# %%
evaluate_longbench_sample()

# %%
