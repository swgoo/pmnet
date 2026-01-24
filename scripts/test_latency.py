import numpy as np
import matplotlib.pyplot as plt


def plot_three_model_flops():
    # 1. 시퀀스 길이 설정 (2k ~ 1M) - 로그 스케일
    seq_lens = np.geomspace(2048, 1048576, num=100)  # 1M까지 확장해서 차이를 더 명확히

    # 2. 모델 하이퍼파라미터 가정 (비슷한 체급 가정)
    # 125M~350M 수준의 Small Model 기준
    d_model = 768  # Hidden Dimension
    n_layers = 24  # Number of Layers

    # PMNet Specifics
    window_size = 2048  # SWA Window
    num_slots = 340  # Memory Slots

    # Mamba Specifics (Mamba-1 Architecture)
    # Expansion Factor E=2, State Dimension N=16
    d_state = 16
    expand = 2
    d_inner = expand * d_model

    # -------------------------------------------------------
    # 3. FLOPs 계산 공식 (Total FLOPs over all layers)
    # -------------------------------------------------------

    # [A] Transformer (Attention + FFN)
    # Linear Part: Projections (Q,K,V,O) + FFN (up/down) ≈ 24 * d^2 (per token)
    # Quadratic Part: Attention Score & Update ≈ 4 * L * d (per token)
    flops_transformer = n_layers * (
        seq_lens * (24 * d_model**2)  # Linear projections
        + (seq_lens**2) * (4 * d_model)  # Quadratic Attention cost
    )

    # [B] PMNet (SWA + Memory R/W + FFN)
    # Linear Part: Projections + FFN ≈ 24 * d^2 (Transformer와 유사하다고 가정)
    # SWA Cost: 2 * W * d (per token)
    # Memory Cost: 4 * N_m * d (Read + Write per token)
    # *Note: Quadratic L^2 항 없음*
    flops_pmnet = n_layers * (
        seq_lens * (24 * d_model**2)  # Basic Projections
        + seq_lens * (2 * window_size * d_model)  # Local Attention
        + seq_lens * (4 * num_slots * d_model)  # Global Memory Access
    )

    # [C] Mamba (SSM + Gating + Projections)
    # Mamba는 Attention이 없고 SSM Scan을 사용.
    # Projections: Input(x, z) -> 2 * (d * 2d) ... 등등
    # 계산 복잡도는 대략 Transformer의 FFN 부분보다 약간 무거운 수준의 Linear Cost.
    # 근사치: ~30 * d^2 (per token) 정도로 잡음 (Quadratic 항 0)
    # Scan Operation은 d_inner * d_state 이므로 매우 가벼움.
    flops_mamba = n_layers * (
        seq_lens
        * (30 * d_model**2)  # Approximate Linear Cost for SSM block
        # Mamba has NO dependency on Window size (W) or Memory slots (Nm)
        # Scan cost is minimal compared to matmuls
    )

    # -------------------------------------------------------
    # 4. 시각화
    # -------------------------------------------------------
    plt.figure(figsize=(12, 7))

    # Transformer (Red)
    plt.plot(
        seq_lens,
        flops_transformer,
        color="firebrick",
        linestyle="--",
        linewidth=2.5,
        label="Transformer ($O(L^2)$)",
    )

    # PMNet (Blue)
    plt.plot(
        seq_lens,
        flops_pmnet,
        color="royalblue",
        linestyle="-",
        linewidth=2.5,
        label="PMNet (Ours, $O(L)$)",
    )

    # Mamba (Green)
    plt.plot(
        seq_lens,
        flops_mamba,
        color="forestgreen",
        linestyle="-.",
        linewidth=2.5,
        label="Mamba ($O(L)$)",
    )

    # Cross-over Point (Transformer vs PMNet)
    crossover_idx = np.argwhere(flops_transformer > flops_pmnet)[0][0]
    plt.scatter(
        [seq_lens[crossover_idx]],
        [flops_transformer[crossover_idx]],
        color="black",
        zorder=5,
    )
    plt.text(
        seq_lens[crossover_idx],
        flops_transformer[crossover_idx] * 2,
        f" Cross-over\n ~{int(seq_lens[crossover_idx])} tokens",
        ha="right",
        fontsize=10,
    )

    # OOM Line
    plt.axvline(x=32768, color="gray", linestyle=":", alpha=0.8)
    plt.text(
        32768,
        plt.ylim()[0] * 10,
        " GPU Memory Limit (Approx) ",
        rotation=90,
        color="gray",
        ha="right",
    )

    plt.title(
        "Theoretical Computational Complexity (FLOPs)\nLinear vs Quadratic Architectures",
        fontsize=16,
    )
    plt.xlabel("Sequence Length (L)", fontsize=14)
    plt.ylabel("Floating Point Operations (FLOPs)", fontsize=14)
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12, loc="upper left")

    save_path = "pmnet_mamba_transformer_complexity.png"
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")


if __name__ == "__main__":
    plot_three_model_flops()
