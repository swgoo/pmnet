# %%
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# === 설정 ===
CKPT_PATH = "HuggingFaceTB/SmolLM-135M"
DATASET_ID = "emozilla/pg19"
DATA_SPLIT = "test"
SEQ_LENGTH = 16384  # 토큰 기준 길이 (충분히 길게 설정)
CHUNK_SIZE = 512  # VRAM 절약용 청크 사이즈

loss_save_dir = "../data/smollm_losses_longctx"
os.makedirs(loss_save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === 모델 및 토크나이저 로드 ===
tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
config = AutoConfig.from_pretrained(CKPT_PATH)

# 순정 상태 로드 (RoPE Scaling 없이 원래 한계 확인용)
model = AutoModelForCausalLM.from_pretrained(
    CKPT_PATH,
    config=config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
model.eval()

# 모델의 원래 한계 확인 (보통 2048)
base_ctx_len = getattr(config, "max_position_embeddings", 2048)
print(f"Model Base Context Length: {base_ctx_len}")

# === 데이터셋 준비 ===
dataset = load_dataset(DATASET_ID, split=DATA_SPLIT, streaming=True)


def get_long_sample(min_len=SEQ_LENGTH):
    for sample in dataset:
        tokens = tokenizer(sample["text"], add_special_tokens=False)["input_ids"]
        if len(tokens) >= min_len:
            # 원본 텍스트 보존을 위해 전체 반환 (나중에 자름)
            return tokens[:min_len]
    return None


print("Fetching a long sample...")
long_tokens = get_long_sample()
if long_tokens is None:
    raise ValueError("데이터셋에서 충분히 긴 샘플을 찾지 못했습니다.")

print(f"Sample loaded. Length: {len(long_tokens)} tokens")


# === 평가 함수 (Byte Position 매핑 추가) ===
def evaluate_bpb_streaming(model, tokenizer, input_ids, chunk_size=512):
    """
    KV Cache를 유지하며 Loss를 계산하고,
    각 토큰의 실제 바이트 길이를 기반으로 Byte Position과 BPB를 반환합니다.
    """
    input_tensor = torch.tensor(
        input_ids, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    seq_len = input_tensor.size(1)

    nlls = []  # Negative Log Likelihood
    byte_indices = []  # Cumulative Byte Position
    byte_lengths = []  # Length of each token in bytes

    current_byte_pos = 0
    past_key_values = None

    criterion = nn.CrossEntropyLoss(reduction="none")

    # [Pre-computation] 토큰별 바이트 길이 미리 계산
    # (주의: BPE 토크나이저는 앞에 공백이 붙는 경우가 있어 decode시 주의 필요)
    # 여기서는 개별 디코딩으로 길이를 잰다 (속도는 느리지만 정확함)
    print("Pre-calculating byte lengths for mapping...")
    decoded_tokens = [len(tokenizer.decode([t]).encode("utf-8")) for t in input_ids]

    # 첫 토큰은 예측 대상이 아니므로 제외 (input_ids[1:] 가 라벨이 됨)
    # Loss는 [0]번째 토큰을 보고 [1]번째를 맞출 때 발생하므로,
    # Loss[i]는 input_ids[i+1]에 대한 Loss임.
    # 따라서 바이트 매핑도 input_ids[1:]부터 해야 함.
    target_byte_lengths = decoded_tokens[1:]

    pbar = tqdm(range(0, seq_len, chunk_size), desc="Evaluating")

    with torch.no_grad():
        loss_buffer_idx = 0  # 전체 시퀀스에서의 Loss 인덱스 트래킹

        for i in pbar:
            end_loc = min(i + chunk_size, seq_len)

            input_chunk = input_tensor[:, i:end_loc]

            # Forward
            outputs = model(
                input_chunk, past_key_values=past_key_values, use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Shift for Causal LM
            # Logit[t] -> Label[t+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_chunk[..., 1:].contiguous()

            if shift_labels.size(1) > 0:
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                chunk_nlls = loss.float().cpu().numpy().tolist()
                nlls.extend(chunk_nlls)

                # 바이트 위치 매핑
                # 이번 청크에서 계산된 Loss 개수만큼 바이트 길이 가져오기
                num_losses = len(chunk_nlls)
                chunk_byte_lens = target_byte_lengths[
                    loss_buffer_idx : loss_buffer_idx + num_losses
                ]
                byte_lengths.extend(chunk_byte_lens)

                # 누적 바이트 인덱스 생성
                for b_len in chunk_byte_lens:
                    current_byte_pos += b_len
                    byte_indices.append(current_byte_pos)

                loss_buffer_idx += num_losses

            del outputs, logits, shift_logits, shift_labels
            torch.cuda.empty_cache()

    return byte_indices, nlls, byte_lengths


# === 실행 및 플로팅 ===
byte_indices, nlls, token_byte_lens = evaluate_bpb_streaming(
    model, tokenizer, long_tokens, CHUNK_SIZE
)

# [BPB 계산]
# BPB = NLL / (ln(2) * Byte_Length)
# 어떤 토큰의 길이가 0바이트(특수토큰 등)일 경우 나누기 에러 방지 (최소 1로 보정하거나 제외)
safe_byte_lens = np.array(token_byte_lens)
safe_byte_lens[safe_byte_lens == 0] = 1  # 예외 처리

bpb_values = np.array(nlls) / (np.log(2) * safe_byte_lens)


# 이동 평균 (튀는 값 보정)
def moving_average_with_indices(vals, idxs, n=200):
    if len(vals) < n:
        return vals, idxs
    ret_vals = np.cumsum(vals, dtype=float)
    ret_vals[n:] = ret_vals[n:] - ret_vals[:-n]
    smoothed_vals = ret_vals[n - 1 :] / n
    return smoothed_vals, idxs[n - 1 :]


smoothed_bpb, smoothed_bytes = moving_average_with_indices(
    bpb_values, byte_indices, n=300
)

plt.figure(figsize=(12, 6))
plt.plot(smoothed_bytes, smoothed_bpb, label="SmolLM-135M BPB", color="grey", alpha=0.8)

# Context Limit 표시 (Byte 단위로 대략적 변환 필요)
# 보통 1 token ~= 4 bytes (영어 기준)
approx_byte_limit = base_ctx_len * 4
plt.axvline(
    x=approx_byte_limit,
    color="r",
    linestyle="--",
    label=f"Context Limit (~{base_ctx_len} tokens)",
)

# 그래프 스타일링
plt.title(f"SmolLM-135M Long Context BPB (Byte-Aligned)\nData: PG19 Test")
plt.xlabel("Position in Bytes")
plt.ylabel("Bits Per Byte (BPB)")
plt.legend()
plt.grid(True, alpha=0.3)

# Y축 범위 제한 (너무 튀는 값 제외)
plt.ylim(0.5, 5.0)

plt.show()

# 저장
plt.savefig(f"{loss_save_dir}/smollm_byte_divergence.png")
print(f"Graph saved to {loss_save_dir}/smollm_byte_divergence.png")
# %%
