# %%
from itertools import chain
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from pmnet import PMNetForCausalLM, PMNetConfig

CKPT_PATH = "../ckpts/byte_batch48_28000"
DATA_PROCESS_BATCH_SIZE = 1000
NUM_PROC = 16
SEQ_LENGTH = 512 * 1024
DATASET_ID = "emozilla/pg19"
DATA_SPLIT = "test"
NUM_EVAL_SAMPLES = 4
loss_save_dir = "../data/byte_losses_pg19_512k"

os.makedirs(loss_save_dir, exist_ok=True)

import torch


class ByteTokenizer:
    def __init__(self, special_tokens: dict[str, int] | None = None):
        self.vocab_size = 256
        self.pad_idx = 0
        self.bos_idx = 254
        self.eos_idx = 255
        self.special_tokens = special_tokens or {}

        # EOS 정보 등 표준 인터페이스용 속성 추가
        self.eos_token_id = self.eos_idx
        self.bos_token_id = self.bos_idx

        # UTF-8 안전 범위 검사
        for idx in [self.bos_idx, self.eos_idx] + list(self.special_tokens.values()):
            assert (
                0xF8 <= idx <= 0xFF
            ), f"Special token index {idx}는 UTF-8 안전 범위를 벗어납니다."

    def __call__(self, seqs: list[str] | str, **kwargs):
        """datasets.map의 표준 인터페이스 대응"""
        if isinstance(seqs, str):
            seqs = [seqs]
        return {"input_ids": self.encode(seqs, **kwargs)}

    def encode(self, seqs: list[str], add_bos=False, add_eos=False) -> list[list[int]]:
        total_outputs = []
        for text in seqs:
            # bytes는 그 자체로 [0~255] 정수 리스트처럼 동작합니다.
            text_byte = list(text.encode("utf-8"))

            if add_bos:
                text_byte = [self.bos_idx] + text_byte
            if add_eos:
                text_byte = text_byte + [self.eos_idx]

            total_outputs.append(text_byte)
        return total_outputs

    def decode(self, tokens: list[int] | torch.Tensor, **kwargs):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if "errors" not in kwargs:
            kwargs["errors"] = "ignore"
        return bytes(tokens).decode("utf-8", **kwargs)


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
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy


def get_loss_over_positions(model, input_ids, ablation=False, chunk_size=1024 * 30):
    model.eval()
    device = model.device
    seq_len = len(input_ids)
    all_losses = []

    past_key_values = None

    original_cumsum_config = model.config.memory_cumsum
    if ablation:
        model.config.memory_cumsum = False

    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_input = input_ids[i:end_idx]

            # [수정 1] 레이블용으로 다음 토큰 1개를 더 가져옴 (가능한 경우)
            # label_end_idx는 입력보다 1칸 더 뒤까지 봄
            label_end_idx = min(i + chunk_size + 1, seq_len)
            chunk_labels_ids = input_ids[i + 1 : label_end_idx]  # 정답은 한 칸 뒤부터

            input_tensor = (
                torch.tensor(chunk_input, dtype=torch.long).unsqueeze(0).to(device)
            )

            outputs = model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            if ablation and past_key_values is not None:
                if hasattr(past_key_values, "_memory_states_storage"):
                    for block_idx in past_key_values._memory_states_storage:
                        past_key_values._memory_states_storage[block_idx].zero_()

            logits = outputs.logits

            if len(chunk_labels_ids) == len(chunk_input):
                # 1:1 매칭이 되므로 logits 전체 사용
                shift_logits = logits.contiguous()
                shift_labels = torch.tensor(chunk_labels_ids, dtype=torch.long).to(
                    device
                )
            else:
                # 마지막 청크라서 다음 정답이 없는 경우 (기존 로직)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_tensor[..., 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            all_losses.extend(loss.cpu().numpy().tolist())

            # decode 테스트용 출력 (필요시 주석 해제)
            # pred_tokens = torch.argmax(shift_logits, dim=-1).squeeze(0)
            # decoded_text = tokenizer.decode(pred_tokens)
            # print(f"Decoded Text Chunk [{i}:{end_idx}]: {decoded_text} \n")

            del outputs
            del logits
            torch.cuda.empty_cache()

    # 설정 복구
    model.config.memory_cumsum = original_cumsum_config

    return all_losses


def evaluate_longbench_sample():
    # 1. 설정 및 모델 로드
    model_path = CKPT_PATH  # 체크포인트 경로 입력
    model = PMNetForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="cuda",
        # trust_remote_code=True,
    )
    # config = PMNetConfig.from_pretrained(model_path)
    # model = PMNetForCausalLM(config).to(device)
    param_size = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {param_size:,}")
    print(f"규모 약어: {param_size / 1_000_000_000:.3f}B")

    # 2. LongBench 데이터 로드 (예: gov_report - 매우 긴 문서)
    # Huggingface Datasets가 없다면 pip install datasets
    dataset = test_dataset
    losses_off_list = []
    losses_on_list = []
    dataset_for_eval = dataset.select(range(NUM_EVAL_SAMPLES))
    for i, sample in enumerate(
        tqdm(dataset_for_eval, desc="Evaluating Samples", total=len(dataset_for_eval))
    ):
        if os.path.exists(
            f"{loss_save_dir}/sample_{i}_losses_off.npy"
        ) and os.path.exists(f"{loss_save_dir}/sample_{i}_losses_on.npy"):
            losses_off = np.load(f"{loss_save_dir}/sample_{i}_losses_off.npy")
            losses_on = np.load(f"{loss_save_dir}/sample_{i}_losses_on.npy")
            losses_off_list.append(losses_off)
            losses_on_list.append(losses_on)
            continue
        else:
            losses_on = get_loss_over_positions(
                model, sample["input_ids"], ablation=False
            )
            np.save(f"{loss_save_dir}/sample_{i}_losses_on.npy", np.array(losses_on))
            losses_on_list.append(losses_on)
            losses_off = get_loss_over_positions(
                model, sample["input_ids"], ablation=True
            )
            np.save(f"{loss_save_dir}/sample_{i}_losses_off.npy", np.array(losses_off))
            losses_off_list.append(losses_off)

    losses_off_mean = np.array(losses_off_list).mean(axis=0)
    losses_on_mean = np.array(losses_on_list).mean(axis=0)

    # 5. 결과 시각화
    plt.figure(figsize=(12, 6))

    # Moving Average로 그래프 부드럽게 만들기 (Window 100)
    def moving_average(a, n=1000):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    smooth_on = moving_average(losses_on_mean)
    smooth_off = moving_average(losses_off_mean)

    smooth_on = smooth_on / 0.693  # Byte 단위 bit 보정
    smooth_off = smooth_off / 0.693  # Byte 단위 bit 보정
    # smooth_on = 2 ** (smooth_on * 4)  # word 단위 PPL 보정
    # smooth_off = 2 ** (smooth_off * 4)  # word 단위 PPL 보정

    x = range(len(smooth_on))

    plt.plot(
        x,
        smooth_off,
        label=f"Ablation (No Recurrence + {model.config.sliding_window}WS)",
        color="red",
        alpha=0.5,
        linestyle="--",
    )
    plt.plot(
        x,
        smooth_on,
        label=f"PMNet (Memory + {model.config.sliding_window}WS)",
        color="blue",
        alpha=0.5,
    )

    # plt.axvline(x=256, color="green", linestyle=":", label="256 Tokens")
    # plt.axvline(x=32, color="gray", linestyle=":", label="32 Tokens (WS)")

    plt.title(f"Bit Per Byte Comparison \nSeq Len: {SEQ_LENGTH//1024}k Tokens")
    plt.xlabel("Token Position")
    plt.ylabel("BPB (Bits Per Byte)")
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


# %%
evaluate_longbench_sample()

# %%
