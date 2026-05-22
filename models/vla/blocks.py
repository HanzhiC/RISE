import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(x, positions, max_wavelength=10000):
    """
    Applies RoPE (Rotary Position Embedding) to input tensor.
    Supports both 1D positions [B, L] or [L] and 2D positions [B, 2, L] for images.

    Args:
        x: Input tensor of shape [B, L, H, D] or [B, L, D] where D is the head dimension
        positions: Position indices of shape:
            - [B, L] or [L] for 1D positions
            - [B, 2, L] for 2D positions (y, x coordinates for image patches)
        max_wavelength: Maximum wavelength for RoPE (default: 10000)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    # Handle different input shapes
    original_shape = x.shape
    if x.dim() == 3:
        # [B, L, D] -> [B, L, 1, D]
        x = x.unsqueeze(2)
        squeeze_output = True
    else:
        squeeze_output = False

    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)
    d_half = x.shape[-1] // 2

    # Check if positions are 2D (image coordinates) or 1D
    is_2d_positions = positions.dim() == 3 and positions.shape[1] == 2

    if is_2d_positions:
        # 2D positions: [B, 2, L] where positions[:, 0, :] are y-coords and positions[:, 1, :] are x-coords
        assert (
            positions.shape[0] == x.shape[0]
        ), "Batch size mismatch between x and positions"
        assert (
            positions.shape[2] == x.shape[1]
        ), "Sequence length mismatch between x and positions"

        # Split features into two halves: first half for y-coords, second half for x-coords
        y_features, x_features = x.chunk(
            2, dim=-1
        )  # Each: [B, L, H, D/2] or [B, L, 1, D/2]

        # Extract y and x coordinates
        y_positions = positions[:, 0, :]  # [B, L] - y coordinates
        x_positions = positions[:, 1, :]  # [B, L] - x coordinates

        # For each half (D/2 dimensions), we apply standard RoPE which splits into D/4 and D/4
        d_quarter = d_half // 2  # D/4

        # Compute frequencies for each half (D/4 frequencies)
        freq_exponents = (2.0 / d_quarter) * torch.arange(
            d_quarter, dtype=torch.float32, device=device
        )  # [D/4]
        timescale = max_wavelength**freq_exponents  # [D/4]

        # Apply RoPE to y-coordinates (first half of features)
        y_radians = (
            y_positions[..., None].to(torch.float32) / timescale[None, None, :]
        )  # [B, L, D/4]
        y_radians = y_radians[..., None, :]  # [B, L, 1, D/4]
        y_sin = torch.sin(y_radians)
        y_cos = torch.cos(y_radians)
        y1, y2 = y_features.split(d_quarter, dim=-1)  # Each: [B, L, H, D/4]
        y_rotated = torch.empty_like(y_features)
        y_rotated[..., :d_quarter] = y1 * y_cos - y2 * y_sin
        y_rotated[..., d_quarter:] = y2 * y_cos + y1 * y_sin

        # Apply RoPE to x-coordinates (second half of features)
        x_radians = (
            x_positions[..., None].to(torch.float32) / timescale[None, None, :]
        )  # [B, L, D/4]
        x_radians = x_radians[..., None, :]  # [B, L, 1, D/4]
        x_sin = torch.sin(x_radians)
        x_cos = torch.cos(x_radians)
        x1, x2 = x_features.split(d_quarter, dim=-1)  # Each: [B, L, H, D/4]
        x_rotated = torch.empty_like(x_features)
        x_rotated[..., :d_quarter] = x1 * x_cos - x2 * x_sin
        x_rotated[..., d_quarter:] = x2 * x_cos + x1 * x_sin

        # Concatenate y and x rotated features
        res = torch.cat([y_rotated, x_rotated], dim=-1)

    else:
        # 1D positions: [B, L] or [L]
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(x.shape[0], -1)

        # Compute frequencies
        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
            d_half, dtype=torch.float32, device=device
        )  # [D / 2]
        timescale = max_wavelength**freq_exponents  # [D / 2]
        radians = (
            positions[..., None].to(torch.float32) / timescale[None, None, :]
        )  # [B, L, D / 2]
        radians = radians[..., None, :]  # [B, L, 1, D / 2]

        sin = torch.sin(radians)
        cos = torch.cos(radians)

        # Apply RoPE rotation
        x1, x2 = x.split(d_half, dim=-1)
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin

    res = res.to(dtype)

    if squeeze_output:
        res = res.squeeze(2)  # [B, L, 1, D] -> [B, L, D]

    return res


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.pooling_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=True,
            bias=True,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)

    def forward(self, x):
        assert len(x.shape) == 3, "x should be of shape (B, H, D)"
        pooling_token = self.pooling_token.expand(x.shape[0], -1, -1)
        q = self.norm1(pooling_token)
        k, v = self.to_kv(self.norm2(x)).chunk(2, dim=-1)
        attn_output, _ = self.attn(q, k, v)  # [B, 1, D]
        return attn_output.squeeze(1)  # [B, D]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=None):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :param dtype: Optional dtype for the embedding. If None, uses t.dtype or float32.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        # Use dtype from model parameters if available, otherwise float32
        if dtype is None:
            dtype = torch.float32
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t.to(dtype) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # Get dtype from model parameters to maintain consistency
        # Use the first parameter's dtype, or float32 as fallback
        try:
            param_dtype = next(self.mlp.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size, dtype=param_dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class SkipEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.skip_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.embedder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.skip_linear.weight, std=0.02)
        nn.init.normal_(self.embedder[1].weight, std=0.02)
        nn.init.constant_(self.embedder[1].bias, 0)
        nn.init.zeros_(self.embedder[3].weight)
        nn.init.zeros_(self.embedder[3].bias)

    def forward(self, x):
        return self.skip_linear(x) + self.embedder(x)


class CDiTBlockAbsPos(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            add_bias_kv=True,
            bias=True,
            batch_first=True,
            **block_kwargs,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, c, x_cond):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca_xcond,
            scale_ca_xcond,
            shift_ca_x,
            scale_ca_x,
            gate_ca_x,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(11, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = (
            x
            + gate_ca_x.unsqueeze(1)
            * self.cttn(
                query=modulate(self.norm2(x), shift_ca_x, scale_ca_x),
                key=x_cond_norm,
                value=x_cond_norm,
                need_weights=False,
            )[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Supports optional RoPE (Rotary Position Embedding).
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_rope=True,
        rope_max_wavelength=10000,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope
        self.rope_max_wavelength = rope_max_wavelength

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            add_bias_kv=True,
            bias=True,
            batch_first=True,
            **block_kwargs,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(
        self,
        x,
        c,
        x_cond,
        context,
        x_positions=None,
        x_cond_positions=None,
        context_positions=None,
    ):
        """
        Forward pass with optional RoPE support.

        Args:
            x: Input tensor [B, L_x, D]
            c: Conditioning tensor [B, D]
            x_cond: Condition tensor [B, L_cond, D]
            context: Context tensor [B, L_context, D]
            x_positions: Optional position indices for x [B, 2, L_x]
            x_cond_positions: Optional position indices for x_cond [B, 2, L_cond]
            context_positions: Optional position indices for context [B, L_context]
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca_xcond,
            scale_ca_xcond,
            shift_ca_x,
            scale_ca_x,
            gate_ca_x,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(11, dim=1)

        # Self-attention with optional RoPE
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.use_rope:
            assert x_positions is not None
            assert x_cond_positions is not None
            assert context_positions is not None
            # Manually compute QKV to apply RoPE
            B, L, D = x_norm.shape
            qkv = self.attn.qkv(x_norm)  # [B, L, 3*D]
            qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(
                2, 0, 3, 1, 4
            )  # [3, B, H, L, D]
            q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, L, D]
            # Generate positions if not provided
            if x_positions is None:
                x_positions = torch.arange(L, device=x.device, dtype=torch.long)

            # Apply RoPE to Q and K
            q = apply_rope(
                q.permute(0, 2, 1, 3), x_positions, self.rope_max_wavelength
            ).permute(
                0, 2, 1, 3
            )  # [B, H, L, D]
            k = apply_rope(
                k.permute(0, 2, 1, 3), x_positions, self.rope_max_wavelength
            ).permute(
                0, 2, 1, 3
            )  # [B, H, L, D]

            # Compute attention using scaled_dot_product_attention
            dropout_p = (
                self.attn.attn_drop.p if hasattr(self.attn.attn_drop, "p") else 0.0
            )
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=dropout_p if self.training else 0.0,
                scale=self.head_dim**-0.5,
            )  # [B, H, L, D]
            attn_output = attn_output.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
            attn_output = self.attn.proj(attn_output)
            attn_output = self.attn.proj_drop(attn_output)
        else:
            attn_output = self.attn(x_norm)

        x = x + gate_msa.unsqueeze(1) * attn_output

        # Cross-attention with optional RoPE
        L_context = context.shape[1]
        x_cond_context = torch.cat([x_cond, context], dim=1)
        x_cond_context_norm = modulate(
            self.norm_cond(x_cond_context), shift_ca_xcond, scale_ca_xcond
        )
        x_query = modulate(self.norm2(x), shift_ca_x, scale_ca_x)

        if self.use_rope:
            assert x_positions is not None
            assert x_cond_positions is not None
            assert context_positions is not None
            # For cross-attention, we need to apply RoPE to query and key separately
            # Extract Q, K, V from MultiheadAttention
            B_x, L_x, D = x_query.shape
            B_cond, L_cond, D = x_cond_context_norm.shape

            # Get Q, K, V projections from MultiheadAttention
            # Note: MultiheadAttention uses in_proj_weight and in_proj_bias
            # We'll need to manually compute Q, K, V to apply RoPE
            q = F.linear(
                x_query,
                self.cttn.in_proj_weight[: self.hidden_size],
                (
                    self.cttn.in_proj_bias[: self.hidden_size]
                    if self.cttn.in_proj_bias is not None
                    else None
                ),
            )
            k = F.linear(
                x_cond_context_norm,
                self.cttn.in_proj_weight[self.hidden_size : 2 * self.hidden_size],
                (
                    self.cttn.in_proj_bias[self.hidden_size : 2 * self.hidden_size]
                    if self.cttn.in_proj_bias is not None
                    else None
                ),
            )
            v = F.linear(
                x_cond_context_norm,
                self.cttn.in_proj_weight[2 * self.hidden_size :],
                (
                    self.cttn.in_proj_bias[2 * self.hidden_size :]
                    if self.cttn.in_proj_bias is not None
                    else None
                ),
            )

            # Reshape to [B, L, H, D]
            q = q.reshape(B_x, L_x, self.num_heads, self.head_dim)
            k = k.reshape(B_cond, L_cond, self.num_heads, self.head_dim)
            v = v.reshape(B_cond, L_cond, self.num_heads, self.head_dim)

            # Apply RoPE to Q and K
            q = apply_rope(q, x_positions, self.rope_max_wavelength)  # [B, L_x, H, D]
            # k = apply_rope(
            #     k, context_positions, self.rope_max_wavelength
            # )  # [B, L_context, H, D]
            k_cond, k_context = (
                k[:, : L_cond - L_context, :, :],
                k[:, L_cond - L_context :, :, :],
            )
            assert (
                k_context.shape[1] == L_context
                and k_cond.shape[1] == L_cond - L_context
            )
            k_cond = apply_rope(k_cond, x_cond_positions, self.rope_max_wavelength)
            k_context = apply_rope(
                k_context, context_positions, self.rope_max_wavelength
            )
            k = torch.cat([k_cond, k_context], dim=1)

            # Reshape for attention: [B, H, L, D]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            # Handle bias_kv if present (for add_bias_kv=True)
            # bias_k has shape [1, 1, embed_dim] = [1, 1, H * head_dim]
            # Need to reshape to [1, 1, H, head_dim] to match k shape [B, H, L, head_dim]
            if hasattr(self.cttn, "bias_k") and self.cttn.bias_k is not None:
                bias_k = self.cttn.bias_k  # [1, 1, embed_dim]
                bias_k = bias_k.reshape(1, 1, self.num_heads, self.head_dim).permute(
                    0, 2, 1, 3
                )  # [1, H, 1, head_dim]
                k = (
                    k + bias_k
                )  # Broadcasting: [B, H, L, head_dim] + [1, 1, H, head_dim]
            if hasattr(self.cttn, "bias_v") and self.cttn.bias_v is not None:
                bias_v = self.cttn.bias_v  # [1, 1, embed_dim]
                bias_v = bias_v.reshape(1, 1, self.num_heads, self.head_dim).permute(
                    0, 2, 1, 3
                )  # [1, H, 1, head_dim]
                v = (
                    v + bias_v
                )  # Broadcasting: [B, H, L, head_dim] + [1, H, 1, head_dim]

            # Compute cross-attention using scaled_dot_product_attention
            dropout_p = getattr(self.cttn, "dropout", 0.0)
            if isinstance(dropout_p, nn.Module):
                dropout_p = dropout_p.p
            dropout_p = dropout_p if isinstance(dropout_p, (int, float)) else 0.0
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=dropout_p if self.training else 0.0,
                scale=self.head_dim**-0.5,
            )  # [B_x, H, L_x, head_dim]
            attn_output = attn_output.transpose(1, 2).reshape(
                B_x, L_x, D
            )  # [B, L_x, D]

            # Apply output projection
            attn_output = F.linear(
                attn_output, self.cttn.out_proj.weight, self.cttn.out_proj.bias
            )
        else:
            attn_output = self.cttn(
                query=x_query,
                key=x_cond_context_norm,
                value=x_cond_context_norm,
                need_weights=False,
            )[0]

        x = x + gate_ca_x.unsqueeze(1) * attn_output

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x
