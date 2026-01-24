# %%
from itertools import chain
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from pmnet import PMNetForCausalLM, PMNetConfig, ByteTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

# ==========================================
# [설정] 경로 및 하이퍼파라미터
# ==========================================
CKPT_PATH = "../ckpts/byte_batch48_28000"
DATA_PROCESS_BATCH_SIZE = 1000
NUM_PROC = 16
SEQ_LENGTH = 128 * 1024
DATASET_ID = "emozilla/pg19"
DATA_SPLIT = "test"
NUM_EVAL_SAMPLES = 40
loss_save_dir = "../data/byte_losses_pg19_128k_hierarchy_test"  # 저장 경로 변경
os.makedirs(loss_save_dir, exist_ok=True)

# ==========================================
# [데이터] 로드 및 전처리
# ==========================================
tokenizer = ByteTokenizer()
test_dataset = load_dataset(DATASET_ID, split=DATA_SPLIT)
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_batch(examples):
    tokenized = tokenizer(examples["text"])
    input_ids = tokenized["input_ids"]
    # SEQ_LENGTH 만큼 자름
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


# ==========================================
# [함수] Loss 계산 (순수 Inference)
# ==========================================
def get_loss_over_positions(model, input_ids, chunk_size=1024 * 30):
    """
    모델의 현재 Weight 상태 그대로 Inference를 수행하여 Loss를 계산함.
    (내부적인 Ablation 로직 제거됨 - 외부에서 Weight 조작 후 호출)
    """
    model.eval()
    device = model.device
    seq_len = len(input_ids)
    all_losses = []
    past_key_values = None
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_input = input_ids[i:end_idx]

            # 레이블용: 다음 토큰 1개를 더 가져옴
            label_end_idx = min(i + chunk_size + 1, seq_len)
            chunk_labels_ids = input_ids[i + 1 : label_end_idx]

            input_tensor = (
                torch.tensor(chunk_input, dtype=torch.long).unsqueeze(0).to(device)
            )

            outputs = model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits

            if len(chunk_labels_ids) == len(chunk_input):
                shift_logits = logits.contiguous()
                shift_labels = torch.tensor(chunk_labels_ids, dtype=torch.long).to(
                    device
                )
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_tensor[..., 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            all_losses.extend(loss.cpu().numpy().tolist())

            del outputs, logits
            torch.cuda.empty_cache()

    return all_losses


# ==========================================
# [함수] 메인 실험 루프 (Hierarchy & Drift)
# ==========================================
def evaluate_hierarchy_ablation():
    # 1. 모델 로드
    print(f"Loading Model from {CKPT_PATH}...")
    model = PMNetForCausalLM.from_pretrained(
        CKPT_PATH,
        dtype=torch.float32,  # 안정성을 위해 float32 추천, 메모리 부족시 bfloat16
        device_map="cuda",
    )
    param_size = sum(p.numel() for p in model.parameters())
    print(f"Model Size: {param_size / 1_000_000_000:.3f}B")

    # 2. 원본 임베딩 백업 (Weight Restoration용)
    # Deepcopy는 안전하지만 느릴 수 있으므로 텐서만 Clone해서 리스트로 저장
    original_embeddings = [
        emb.clone().detach() for emb in model.model.memory_embeddings
    ]

    # 3. 실험 모드 정의
    # (Mode Name, File Suffix, Color, Label)
    modes = [
        ("Normal", "normal", "blue", "Normal (Baseline)"),
        ("Leaf Zero", "leaf_zero", "green", "Leaf Zero (Detail Ablation)"),
        ("Root Zero", "root_zero", "red", "Root Zero (Structure Ablation)"),
        ("All Zero", "all_zero", "orange", "All Zero (NTM Drift Mode)"),
    ]

    dataset = test_dataset.select(range(NUM_EVAL_SAMPLES))

    # 결과 저장용 딕셔너리
    results = {mode_name: [] for mode_name, _, _, _ in modes}

    for i, sample in enumerate(tqdm(dataset, desc="Samples", total=len(dataset))):

        for mode_name, suffix, _, _ in modes:
            save_file = f"{loss_save_dir}/sample_{i}_losses_{suffix}.npy"

            # 이미 계산된 결과가 있으면 로드
            if os.path.exists(save_file):
                losses = np.load(save_file)
                results[mode_name].append(losses)
                continue

            # --- [핵심] Weight Manipulation Logic ---
            # 1. 먼저 원본으로 복구 (Reset)
            with torch.no_grad():
                for idx, emb in enumerate(model.model.memory_embeddings):
                    emb.data.copy_(original_embeddings[idx])

            # 2. 모드별로 임베딩 0으로 밀기 (Ablation Apply)
            with torch.no_grad():
                if mode_name == "Normal":
                    pass  # 아무것도 안함

                elif mode_name == "All Zero":
                    # 모든 닻을 제거 -> Drift 유발
                    for emb in model.model.memory_embeddings:
                        emb.data.fill_(0.0)

                elif mode_name == "Root Zero":
                    # 최상위(Root) 계층 파괴 -> 구조 붕괴
                    # *사용자의 메모리 구조상 index 0이 Root라고 가정*
                    model.model.memory_embeddings[0].data.fill_(0.0)

                elif mode_name == "Leaf Zero":
                    # 최하위(Leaf) 계층 파괴 -> 디테일 손실
                    model.model.memory_embeddings[-1].data.fill_(0.0)

            # 3. Inference 수행
            losses = get_loss_over_positions(model, sample["input_ids"])
            np.save(save_file, np.array(losses))
            results[mode_name].append(losses)

    # 4. 결과 시각화 (Delta BPB 분석)
    plt.figure(figsize=(12, 7))

    # Moving Average 함수 (윈도우를 크게 잡아야 경향성이 잘 보입니다)
    def moving_average(a, n=10000):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    # 기준점(Normal) 계산
    normal_losses = np.array(results["Normal"]).mean(axis=0)
    normal_bpb = moving_average(normal_losses) / 0.693

    # 각 모드별 Delta 계산 및 플로팅
    for mode_name, suffix, color, label in modes:
        loss_list = results[mode_name]
        if not loss_list:
            continue

        # 샘플별 평균 계산
        avg_losses = np.array(loss_list).mean(axis=0)
        current_bpb = moving_average(avg_losses) / 0.693

        # Delta 계산: (현재 모드 BPB) - (Normal 모드 BPB)
        # Normal은 자기 자신을 빼므로 정확히 0이 됩니다.
        delta_bpb = current_bpb - normal_bpb

        x_axis = range(len(delta_bpb))
        cumulative_delta = np.cumsum(delta_bpb)
        # 그래프 스타일
        if mode_name == "Normal":

            plt.plot(
                x_axis,
                cumulative_delta,
                label=label,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
        else:
            plt.plot(
                x_axis,
                cumulative_delta,
                label=f"Δ {label}",
                color=color,
                linewidth=2,
                alpha=0.9,
            )

    plt.title(
        f"Memory Anchor Ablation: Delta BPB Analysis\n(Relative to Normal Baseline, Seq Len: {SEQ_LENGTH//1024}k)",
        fontsize=14,
    )
    plt.xlabel("Token Position", fontsize=12)
    plt.ylabel("Increase in BPB (Loss Gap)", fontsize=12)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.2)  # 0선 강조
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Drift를 강조하고 싶다면 뒤쪽 20% 구간을 확대해서 보여주는 inset을 만들거나,
    # y축 범위를 타이트하게 조정하세요.
    # plt.ylim(-0.0001, 0.001)

    save_path = "pmnet_delta_bpb_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"Delta graph saved to {save_path}")

    # 5. 수치 출력
    print("\n=== Final Results (Avg BPB) ===")
    for mode_name, _, _, _ in modes:
        if results[mode_name]:
            avg_all_loss = np.mean([np.mean(l) for l in results[mode_name]])
            print(f"{mode_name:10s}: {avg_all_loss / 0.693:.4f} BPB")


# %%
if __name__ == "__main__":
    evaluate_hierarchy_ablation()
# %%
