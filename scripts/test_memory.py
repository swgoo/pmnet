# %%
from itertools import chain
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# from pmnet import PMNetForCausalLM, PMNetConfig, ByteTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

CKPT_PATH = "pmnet-icml2026-10724/pmnet"
DATA_PROCESS_BATCH_SIZE = 1000
NUM_PROC = 16
SEQ_LENGTH = 128 * 1024
DATASET_ID = "emozilla/pg19"
DATA_SPLIT = "test"
NUM_EVAL_SAMPLES = 30
loss_save_dir = "data/byte_losses_pg19_128k_hierarchy_test"  # 저장 경로 변경
os.makedirs(loss_save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
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


def get_loss_over_positions(model, input_ids, chunk_size=1024 * 30):
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


def evaluate_hierarchy_ablation():
    print(f"Loading Model from {CKPT_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_PATH,
        dtype=torch.float32,
        device_map="cuda",
        trust_remote_code=True,
    )
    param_size = sum(p.numel() for p in model.parameters())
    print(f"Model Size: {param_size / 1_000_000_000:.3f}B")

    original_embeddings = [
        emb.clone().detach() for emb in model.model.memory_embeddings
    ]

    modes = [
        ("Normal", "normal", "blue", "Normal (Baseline)"),
        ("No Leaf Memory Embeddings", "leaf_zero", "green", "No Leaf Memory Embeddings"),
        ("No Root Memory Embeddings", "root_zero", "red", "No Root Memory Embeddings"),
        ("No Memory Embeddings", "all_zero", "orange", "No Memory Embeddings"),
    ]

    dataset = test_dataset.select(range(NUM_EVAL_SAMPLES))

    # 결과 저장용 딕셔너리
    results = {mode_name: [] for mode_name, _, _, _ in modes}

    for i, sample in enumerate(tqdm(dataset, desc="Samples", total=len(dataset))):

        for mode_name, suffix, _, _ in modes:
            save_file = f"{loss_save_dir}/sample_{i}_losses_{suffix}.npy"

            if os.path.exists(save_file):
                losses = np.load(save_file)
                results[mode_name].append(losses)
                continue

            with torch.no_grad():
                for idx, emb in enumerate(model.model.memory_embeddings):
                    emb.data.copy_(original_embeddings[idx])

            with torch.no_grad():
                if mode_name == "Normal":
                    pass

                elif mode_name == "No Memory Embeddings":
                    for emb in model.model.memory_embeddings:
                        emb.data.fill_(0.0)

                elif mode_name == "No Root Memory Embeddings":
                    model.model.memory_embeddings[0].data.fill_(0.0)

                elif mode_name == "No Leaf Memory Embeddings":
                    model.model.memory_embeddings[-1].data.fill_(0.0)

            losses = get_loss_over_positions(model, sample["input_ids"])
            np.save(save_file, np.array(losses))
            results[mode_name].append(losses)

    plt.figure(figsize=(8, 6))

    def moving_average(a, n=10000):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    normal_losses = np.array(results["Normal"]).mean(axis=0)
    normal_bpb = moving_average(normal_losses) / 0.693

    for mode_name, suffix, color, label in modes:
        loss_list = results[mode_name]
        if not loss_list:
            continue

        avg_losses = np.array(loss_list).mean(axis=0)
        current_bpb = moving_average(avg_losses) / 0.693

        delta_bpb = current_bpb - normal_bpb

        x_axis = range(len(delta_bpb))
        cumulative_delta = np.cumsum(delta_bpb)
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
        f"Cumulative Delta BPB relative to Baseline\n Seq Len: {SEQ_LENGTH//1024}k Tokens",
        fontsize=20,
        fontweight='bold'
    )
    plt.xlabel("Token Position", fontsize=20)
    plt.ylabel("Cumulative BPB Degradation (Bits)", fontsize=20)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.2)  # 0선 강조
    plt.legend(loc="upper left", fontsize=20)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)


    save_path = "pmnet_delta_bpb_analysis.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Delta graph saved to {save_path}")

    print("\n=== Final Results (Avg BPB) ===")
    for mode_name, _, _, _ in modes:
        if results[mode_name]:
            avg_all_loss = np.mean([np.mean(l) for l in results[mode_name]])
            print(f"{mode_name:10s}: {avg_all_loss / 0.693:.4f} BPB")


# %%
if __name__ == "__main__":
    evaluate_hierarchy_ablation()
# %%
