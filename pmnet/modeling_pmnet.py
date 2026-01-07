from collections.abc import Callable
from typing import Iterable, Optional, Union

from einops import einsum, rearrange
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from .configuration_pmnet import PMNetConfig


class PMNetCache(DynamicCache):

    def __init__(
        self,
        ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
        config: Optional[PMNetConfig] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
        self.config = config
        self._memory_states_storage: dict[int, torch.Tensor] = {}

    def _get_memory_block_idx(self, layer_idx: int) -> int:
        return (
            layer_idx // self.config.memory_write_period
        ) * self.config.memory_write_period

    def _get_num_memory_groups(self, layer_idx: int) -> int:
        write_step = layer_idx // self.config.memory_write_period
        return self.config.num_memory**write_step

    def get_memory_state(
        self,
        layer_idx: int,
        batch_indices: torch.LongTensor,
        memory_group_indices: torch.LongTensor,
    ) -> torch.Tensor | None:
        assert batch_indices.size(0) == memory_group_indices.size(
            0
        ), "Batch indices and memory group indices must have the same length."

        block_idx = self._get_memory_block_idx(layer_idx)
        if block_idx in self._memory_states_storage:
            storage = self._memory_states_storage[block_idx]
            return storage[batch_indices, memory_group_indices]
        else:
            return torch.zeros(
                batch_indices.size(0),
                self.config.num_memory,
                self.config.memory_size,
                device=batch_indices.device,
                dtype=torch.float32,
            )

    def update_memory_state(
        self,
        layer_idx: int,
        batch_indices: torch.LongTensor,
        memory_group_indices: torch.LongTensor,
        new_state: torch.Tensor,
    ):
        assert batch_indices.size(0) == memory_group_indices.size(
            0
        ), "Batch indices and memory group indices must have the same length."

        block_idx = self._get_memory_block_idx(layer_idx)
        storage = (
            self._memory_states_storage[block_idx]
            if block_idx in self._memory_states_storage
            else None
        )
        if storage is None:
            storage = torch.zeros(
                batch_indices.max().item() + 1,
                self._get_num_memory_groups(layer_idx),
                self.config.num_memory,
                self.config.memory_size,
                device=new_state.device,
                dtype=torch.float32,
            )
            self._memory_states_storage[block_idx] = storage

        if storage.shape[0] < batch_indices.max().item() + 1:
            pad_size = batch_indices.max().item() + 1 - storage.shape[0]
            pad_tensor = torch.zeros(
                pad_size,
                *storage.shape[1:],
                device=storage.device,
                dtype=torch.float32,
            )
            storage = torch.cat([storage, pad_tensor], dim=0)
            self._memory_states_storage[block_idx] = storage

        storage[batch_indices, memory_group_indices] = new_state.to(storage.dtype)


class PMNetRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PMNetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class PMNetRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: PMNetConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class PMNetAttention(nn.Module):

    def __init__(self, config: PMNetConfig, layer_idx: int):
        super().__init__()
        self.layer_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        )
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = PMNetRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = PMNetRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PMNetMemoryWriteModule(nn.Module):
    def __init__(self, config: PMNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.num_memory = config.num_memory
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.num_memory_groups_in_layer = self.num_memory ** (
            layer_idx // config.memory_write_period
        )

        self.norm = PMNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.proj_q = nn.Linear(self.hidden_size, self.memory_size)
        self.proj_k = nn.Linear(2 * self.memory_size, self.memory_size)
        self.proj_v = nn.Linear(self.hidden_size, self.memory_size)
        self.proj_out = nn.Linear(self.memory_size, self.memory_size)

        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        active_memory_embeddings: torch.Tensor,
        memory_group_indices: torch.Tensor,
        cache_params: Optional[PMNetCache] = None,
    ):
        dtype_original = hidden_states.dtype
        device = hidden_states.device
        B, S, _ = hidden_states.shape

        hs = self.norm(hidden_states)
        q = self.proj_q(hs).to(torch.float32).unsqueeze(-2).tanh() * torch.pi
        v = self.proj_v(hs).to(torch.float32)

        memory_angle = memory_states + active_memory_embeddings
        memory_geo = torch.cat([memory_angle.sin(), memory_angle.cos()], dim=-1)
        k = self.proj_k(memory_geo).to(torch.float32).tanh() * torch.pi

        scores = ((q - k).cos().sum(-1) / self.memory_size**0.5).softmax(dim=-1)

        _, indices = torch.topk(scores, k=1, dim=-1)
        next_group_indices = (memory_group_indices * self.num_memory) + indices.squeeze(
            -1
        )

        theta = self.proj_out(v).tanh() * torch.pi
        theta_grid = einsum(theta, scores, "... s m, ... s n -> ... s n m")

        if not self.config.memory_cumsum:
            return theta_grid.to(dtype_original), next_group_indices

        P = B * S
        flat_group = memory_group_indices.reshape(-1).to(torch.long)  # [B*S]
        flat_batch = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(
            S
        )  # [B*S]
        sort_key = flat_batch * self.num_memory_groups_in_layer + flat_group
        perm = torch.argsort(sort_key, stable=True)  # [B*S]
        key_sorted = sort_key[perm]

        is_change = torch.zeros(P, device=device, dtype=torch.bool)
        is_change[0] = True
        is_change[1:] = key_sorted[1:] != key_sorted[:-1]

        start_pos = torch.nonzero(is_change, as_tuple=True)[0]

        ends = torch.cat([start_pos, torch.tensor([P], device=device)])
        seg_lens = ends[1:] - ends[:-1]  # [K]

        counts_sorted = seg_lens.repeat_interleave(seg_lens)

        if self.training and theta_grid.requires_grad:
            counts_flat = torch.empty(P, device=device, dtype=torch.long)
            counts_flat[perm] = counts_sorted
            scale = (
                counts_flat.view(B, S, 1, 1).to(theta_grid.dtype).sqrt().clamp(min=1.0)
            )

            theta_grid = theta_grid / scale + (theta_grid - theta_grid / scale).detach()

        delta_flat = theta_grid.reshape(P, self.num_memory, self.memory_size)
        delta_sorted = delta_flat[perm]  # [P, N, M]

        global_cumsum = torch.cumsum(delta_sorted, dim=0)

        c_shifted = torch.cat(
            [
                torch.zeros(
                    1,
                    self.num_memory,
                    self.memory_size,
                    device=device,
                    dtype=global_cumsum.dtype,
                ),
                global_cumsum[:-1],
            ],
            dim=0,
        )

        offsets = c_shifted[start_pos]
        offsets_expanded = offsets.repeat_interleave(seg_lens, dim=0)

        seg_cumsum_sorted = global_cumsum - offsets_expanded

        if cache_params is not None:
            seg_keys = key_sorted[start_pos]
            seg_batch = (seg_keys // self.num_memory_groups_in_layer).to(torch.long)
            seg_group = (seg_keys % self.num_memory_groups_in_layer).to(torch.long)

            prev_seg_state = cache_params.get_memory_state(
                self.layer_idx, seg_batch, seg_group
            )

            prev_expanded = prev_seg_state.repeat_interleave(seg_lens, dim=0)
            seg_cumsum_sorted = seg_cumsum_sorted + prev_expanded

            end_pos = torch.cat(
                [start_pos[1:] - 1, torch.tensor([P - 1], device=device)]
            )
            new_seg_state = seg_cumsum_sorted[end_pos]
            new_seg_state = torch.remainder(new_seg_state, 2 * torch.pi)

            cache_params.update_memory_state(
                self.layer_idx, seg_batch, seg_group, new_seg_state
            )

        read_state_flat = torch.empty_like(seg_cumsum_sorted)
        read_state_flat[perm] = seg_cumsum_sorted
        read_state = read_state_flat.view(B, S, self.num_memory, self.memory_size)
        read_state = torch.remainder(read_state, 2 * torch.pi)
        return read_state.to(dtype_original), next_group_indices


class PMNetMemoryReadModule(nn.Module):
    def __init__(
        self,
        config: PMNetConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.memory_size = config.memory_size
        self.num_memory_read_heads = config.num_memory_read_heads
        self.layer_idx = layer_idx

        self.norm = PMNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj_q = nn.Linear(
            self.hidden_size, self.num_memory_read_heads * self.memory_size
        )
        self.proj_kv = nn.Linear(
            2 * self.memory_size, 2 * self.num_memory_read_heads * self.memory_size
        )
        self.proj_out = nn.Linear(
            self.num_memory_read_heads * self.memory_size, self.hidden_size
        )
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        active_embeddings: torch.Tensor,
    ):
        """
        hidden_states: [..., hidden_size]
        memory_states: [..., num_memory, memory_size]
        active_embeddings: [..., num_memory, memory_size]
        """

        q = self.proj_q(self.norm(hidden_states)).to(torch.float32).tanh() * torch.pi
        q = rearrange(q, "...  (h m) -> ... h 1 m", h=self.num_memory_read_heads)

        memory_angle = memory_states + active_embeddings
        memory_geo = torch.cat([memory_angle.sin(), memory_angle.cos()], dim=-1)
        kv = self.proj_kv(memory_geo).to(torch.float32)
        kv = rearrange(
            kv, "...  n (two h m) -> ... two h n m", two=2, h=self.num_memory_read_heads
        )
        k, v = kv[..., 0, :, :, :].tanh() * torch.pi, kv[..., 1, :, :, :]

        scores = ((q - k).cos().sum(-1) / self.memory_size**0.5).softmax(dim=-1)

        out = einsum(scores, v, "... h n, ... h n m -> ... h m")
        out = rearrange(out, "... h m -> ... (h m)")
        return self.proj_out(out)


class PMNetDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PMNetConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = PMNetAttention(config=config, layer_idx=layer_idx)
        self.mlp = PMNetMLP(config)
        self.input_layernorm = PMNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PMNetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

        self.memory_read_module = PMNetMemoryReadModule(
            config=config,
            layer_idx=layer_idx,
        )
        if layer_idx % config.memory_write_period == 0:
            self.memory_write_module = PMNetMemoryWriteModule(
                config=config,
                layer_idx=layer_idx,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        active_memory_embeddings: torch.Tensor,
        memory_group_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PMNetCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        next_memory_group_indices = None
        # Memory Write
        if hasattr(self, "memory_write_module"):
            delta_memory, next_memory_group_indices = self.memory_write_module(
                hidden_states,
                memory_states,
                active_memory_embeddings,
                memory_group_indices,
                cache_params=past_key_values,
            )
            memory_states = memory_states + delta_memory

        # Memory Read
        memory_read_out = self.memory_read_module(
            hidden_states,
            memory_states,
            active_memory_embeddings,
        )
        hidden_states = hidden_states + memory_read_out

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, memory_states, next_memory_group_indices


class PMNetPreTrainedModel(PreTrainedModel):
    config: PMNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PMNetDecoderLayer,
        "attentions": PMNetAttention,
    }


class PMNetModel(PMNetPreTrainedModel):
    def __init__(self, config: PMNetConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.memory_embeddings = nn.ParameterList()
        current_num_groups = 1
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx % config.memory_write_period == 0:
                emb = nn.Parameter(
                    torch.zeros(
                        current_num_groups,
                        config.num_memory,
                        config.memory_size,
                    )
                )
                nn.init.normal_(emb, mean=0.0, std=0.02)
                self.memory_embeddings.append(emb)
                current_num_groups *= config.num_memory

            layer = PMNetDecoderLayer(
                config=config,
                layer_idx=layer_idx,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = PMNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PMNetRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PMNetCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = PMNetCache(
                config=self.config,
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        batch_size, seq_len, _ = hidden_states.shape
        memory_states = torch.zeros(
            batch_size,
            seq_len,
            self.config.num_memory,
            self.config.memory_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        current_memory_group_indices = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.long,
            device=hidden_states.device,
        )
        next_memory_group_indices = None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if i % self.config.memory_write_period == 0:
                if i > 0 and next_memory_group_indices is not None:
                    current_memory_group_indices = next_memory_group_indices
            memory_emb_idx = i // self.config.memory_write_period
            memory_embeddings_pool = self.memory_embeddings[memory_emb_idx]
            flat_indices = current_memory_group_indices.view(-1)
            active_memory_embeddings = memory_embeddings_pool.index_select(
                0, flat_indices
            )
            active_memory_embeddings = active_memory_embeddings.view(
                batch_size, seq_len, self.config.num_memory, self.config.memory_size
            )

            hidden_states, memory_states, new_next_indices = decoder_layer(
                hidden_states,
                memory_states,
                active_memory_embeddings,
                current_memory_group_indices,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if new_next_indices is not None:
                next_memory_group_indices = new_next_indices

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class PMNetForCausalLM(PMNetPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = PMNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "PMNetForCausalLM",
    "PMNetPreTrainedModel",
    "PMNetModel",
]
