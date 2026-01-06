# %%
from itertools import chain
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from pmnet import PMNetForCausalLM, PMNetConfig

CKPT_PATH = "ckpts/17000"
DATA_PROCESS_BATCH_SIZE = 1000
NUM_PROC = 16
SEQ_LENGTH = 128 * 1024
DATASET_ID = "zai-org/LongBench-v2"
TOKENIZER_ID = "Qwen/qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

test_dataset = load_dataset(DATASET_ID, split="train")

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_batch(examples):
    tokenized = tokenizer(examples["context"])
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
    desc="Tokenizing & Grouping Train",
)

print(f"Number of test samples: {len(test_dataset)}")
# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy


def get_loss_over_positions(model, input_ids, ablation=False, chunk_size=2048):
    model.eval()
    device = model.device
    seq_len = input_ids.size(1)

    # 결과 저장용
    all_losses = []

    # 초기 상태
    past_key_values = None

    # Ablation 모드일 경우: 모델의 memory_cumsum 설정을 잠시 끔
    original_cumsum_config = model.config.memory_cumsum
    if ablation:
        model.config.memory_cumsum = False

    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for i in tqdm(
            range(0, seq_len, chunk_size), desc=f"Eval (Ablation={ablation})"
        ):
            end_idx = min(i + chunk_size, seq_len)
            chunk_input = input_ids[:, i:end_idx].to(device)

            current_past = past_key_values if not ablation else None

            outputs = model(
                input_ids=chunk_input,
                past_key_values=current_past,
                use_cache=True,  # 다음 청크를 위해 캐시는 일단 받아옴
            )

            if ablation and outputs.past_key_values is not None:
                outputs.past_key_values._memory_states_storage = {}
                past_key_values = outputs.past_key_values
            else:
                past_key_values = outputs.past_key_values

            # Loss 계산 (Next Token Prediction)
            # Logits: [1, Chunk_Len, Vocab]
            logits = outputs.logits

            # 현재 청크의 타겟 (다음 토큰)
            # 전체 시퀀스에서 i+1 ~ end_idx+1 까지 가져와야 함
            # 경계 처리를 위해 일단 현재 청크 내부에서 계산 (마지막 토큰 제외)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chunk_input[..., 1:].contiguous()

            # [Batch * (Chunk-1)]
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            all_losses.extend(loss.cpu().numpy().tolist())

            # 메모리 정리
            del outputs
            torch.cuda.empty_cache()

    # 설정 복구
    model.config.memory_cumsum = original_cumsum_config

    return all_losses


def evaluate_longbench_sample():
    # 1. 설정 및 모델 로드
    model_path = "./your_checkpoint_path"  # 체크포인트 경로 입력
    model = PMNetForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. LongBench 데이터 로드 (예: gov_report - 매우 긴 문서)
    # Huggingface Datasets가 없다면 pip install datasets
    dataset = load_dataset(
        "THUDM/LongBench", "gov_report", split="test", streaming=True
    )
    data_sample = next(iter(dataset))  # 첫 번째 샘플 가져오기

    context = data_sample["context"]
    print(f"Sample Length (Chars): {len(context)}")

    # 토크나이징 (길이 제한 없이)
    input_ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=True)
    seq_len = input_ids.size(1)
    print(f"Total Tokens: {seq_len}")

    if seq_len > 15000:
        input_ids = input_ids[:, :15000]  # 너무 길면 자름 (테스트용)
        seq_len = 15000

    # 3. Baseline 평가 (Memory ON)
    print(">>> Running Baseline (Memory ON)...")
    losses_on = get_loss_over_positions(model, input_ids, ablation=False)

    # 4. Ablation 평가 (Memory OFF - Embeddings Only)
    print(">>> Running Ablation (Memory OFF)...")
    losses_off = get_loss_over_positions(model, input_ids, ablation=True)

    # 5. 결과 시각화
    plt.figure(figsize=(12, 6))

    # Moving Average로 그래프 부드럽게 만들기 (Window 100)
    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    smooth_on = moving_average(losses_on)
    smooth_off = moving_average(losses_off)

    x = range(len(smooth_on))

    plt.plot(x, smooth_on, label="PMNet (Memory + 32WS)", color="blue", alpha=0.8)
    plt.plot(
        x,
        smooth_off,
        label="Ablation (No Recurrence + 32WS)",
        color="red",
        alpha=0.8,
        linestyle="--",
    )

    plt.axvline(x=256, color="green", linestyle=":", label="256 Tokens")
    plt.axvline(x=32, color="gray", linestyle=":", label="32 Tokens (WS)")

    plt.title(
        f"Token-wise NLL Loss Comparison (LongBench gov_report)\nSeq Len: {seq_len}"
    )
    plt.xlabel("Token Position")
    plt.ylabel("NLL Loss (Lower is better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = "pmnet_ablation_study.png"
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")

    # 6. 통계 출력
    print(f"Avg Loss (ON): {np.mean(losses_on):.4f}")
    print(f"Avg Loss (OFF): {np.mean(losses_off):.4f}")
    print(f"PPL (ON): {np.exp(np.mean(losses_on)):.2f}")
    print(f"PPL (OFF): {np.exp(np.mean(losses_off)):.2f}")


if __name__ == "__main__":
    evaluate_longbench_sample()
