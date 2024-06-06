from chex import PRNGKey
from jax import Array, lax
import jax.numpy as jnp
import flax.linen as nn
import jax
from jax.nn import initializers, swish
from typing import Optional
from flax.linen import LayerNorm, dot_product_attention, dot_product_attention_weights
from jax.lax import Precision


class TimestepEmbedder(nn.Module):
    """
    Rotational positional encoding for tiemestep.
    TODO can we dedupe the positional encoding code?
    """

    hidden_size: int
    frequency_embedding_size: int

    @nn.compact
    def __call__(self, t: Array) -> Array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb: Array = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.silu,
                nn.Dense(self.hidden_size),
            ]
        )(t_freq)
        return t_emb

    @staticmethod
    def timestep_embedding(t: Array, freq_emb_size: int, max_period=10000):
        """
        Apply RoPE to timestep.
        """
        half = freq_emb_size // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if freq_emb_size % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding


class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    dropout_prob: float

    def setup(self):
        use_cfg_embedding = int(self.dropout_prob > 0)
        self.embedding_table = nn.Embed(
            num_embeddings=self.num_classes + use_cfg_embedding,
            features=self.hidden_size,
        )

    def __call__(self, labels: Array, train: bool, rng: PRNGKey, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        labels = labels.astype(jnp.int32)

        # drop N labels from the input
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, rng, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

    def token_drop(self, labels: Array, rng: PRNGKey, force_drop_ids=None):
        """
        Randomly drop labels from the input. This is needed
        to support unconditional generation.
        """
        if force_drop_ids is None:
            drop_ids = jax.random.bernoulli(rng, self.dropout_prob, labels.shape)
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels


class Attention(nn.Module):
    dim: int
    n_heads: int
    dtype: jnp.dtype

    def setup(self):
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Dense(
            self.n_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )
        self.wk = nn.Dense(
            self.n_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )
        self.wv = nn.Dense(
            self.n_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )

        self.wo = nn.Dense(self.dim, use_bias=False)

        self.q_norm = LayerNorm(dtype=self.dtype)
        self.k_norm = LayerNorm(dtype=self.dtype)

    @staticmethod
    def reshape_for_broadcast(freqs_cis: Array, x: Array):
        """
        Reshape (..., dim/2) to (..., 1, 1, dim/2)
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.reshape(shape)

    @staticmethod
    def apply_rotary_emb(
        xq: Array, xk: Array, freqs_sin: Array, freqs_cos: Array
    ) -> tuple[Array, Array]:

        # reshape xq and xk to (..., dim/2)
        xq_r, xq_i = jnp.moveaxis(xq.reshape(xq.shape[:-1] + (-1, 2)), -1, 0)
        xk_r, xk_i = jnp.moveaxis(xk.reshape(xk.shape[:-1] + (-1, 2)), -1, 0)

        # reshape freqs_cos and freqs_sin to (..., 1, 1, dim/2) for broadcasting to xq and xk
        freqs_cos = Attention.reshape_for_broadcast(freqs_cos, xq_r)
        freqs_sin = Attention.reshape_for_broadcast(freqs_sin, xq_r)

        # apply rotation
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        # flatten last two dimensions
        xq_out = lax.collapse(jnp.stack([xq_out_r, xq_out_i], axis=-1), 3)
        xk_out = lax.collapse(jnp.stack([xk_out_r, xk_out_i], axis=-1), 3)

        return xq_out, xk_out

    def __call__(self, x: Array, freqs_sin: Array, freqs_cos: Array):
        bsz, seqlen, _ = x.shape

        xq = self.q_norm(self.wq(x))
        xk = self.k_norm(self.wk(x))
        xv = self.wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_sin, freqs_cos)

        attn_output = dot_product_attention(
            xq,
            xk,
            xv,
            deterministic=True,
            force_fp32_for_softmax=True,
            precision=Precision.DEFAULT,
        )

        # squash the head dim
        attn_output = attn_output.reshape(bsz, seqlen, self.dim)
        output = self.wo(attn_output)
        return output


class FeedForward(nn.Module):
    """
    AKA MLP block. Contract from hidden_dim, to dim, and back to hidden_dim.
    """

    dtype: jnp.dtype
    dim: int
    hidden_dim: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float] = None

    def setup(self):

        hidden_dim = int(2 * self.hidden_dim / 3)
        if self.ffn_dim_multiplier:
            hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        hidden_dim = self.multiple_of * (
            (hidden_dim + self.multiple_of - 1) // self.multiple_of
        )

        self.in_layer = nn.Dense(
            features=hidden_dim,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
            dtype=self.dtype,
        )
        self.mid_layer = nn.Dense(
            features=self.dim,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
            dtype=self.dtype,
        )
        self.out_layer = nn.Dense(
            features=hidden_dim,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
            dtype=self.dtype,
        )

    def __call__(self, x):
        x1 = self.in_layer(x)
        x3 = self.out_layer(x)
        # NOTE: mmdit uses SiLU, but this is equivalent
        return self.mid_layer(swish(x1) * x3)


def modulate(x, shift, scale):
    return x * (1 + jnp.expand_dims(scale, 1)) + jnp.expand_dims(shift, 1)


class TransformerBlock(nn.Module):
    layer_id: int
    dim: int
    n_heads: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float]
    norm_eps: float
    dtype: jnp.dtype

    def setup(self):
        self.attention = Attention(dim=self.dim, n_heads=self.n_heads, dtype=self.dtype)
        self.feed_forward = FeedForward(
            dtype=self.dtype,
            dim=self.dim,
            hidden_dim=4 * self.dim,
            multiple_of=self.multiple_of,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
        )
        self.attention_norm = nn.LayerNorm(epsilon=self.norm_eps, dtype=self.dtype)
        self.ffn_norm = nn.LayerNorm(epsilon=self.norm_eps, dtype=self.dtype)

        # modulate attention and ffn according to timestep and label embeddings
        # NOTE this is how the timestep and label embeddings are applied to the model;
        # applies a bias in attention and ffn
        self.adaLN_modulation = nn.Sequential(
            [
                swish,
                nn.Dense(
                    features=6 * self.dim, kernel_init=nn.initializers.xavier_uniform()
                ),
            ]
        )

    def __call__(
        self, x: Array, freqs_sin: Array, freqs_cos: Array, adaln_input=None
    ) -> Array:
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
                self.adaLN_modulation(adaln_input), 6, axis=-1
            )

            attn_out = self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                freqs_sin,
                freqs_cos,
            )
            x = x + jnp.expand_dims(gate_msa, 1) * attn_out

            ff_out = self.feed_forward(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
            x = x + jnp.expand_dims(gate_mlp, 1) * ff_out
        else:
            x = x + self.attention(self.attention_norm(x), freqs_sin, freqs_cos)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


class FinalLayer(nn.Module):
    """
    Convert the transformer patch tokens back to image space, and apply modulation.
    """

    dtype: jnp.dtype
    dim: int
    patch_size: int
    out_channels: int

    def setup(self):
        self.norm_final = nn.LayerNorm(
            epsilon=1e-6, use_bias=False, use_scale=False, dtype=self.dtype
        )
        self.linear = nn.Dense(
            features=self.patch_size * self.patch_size * self.out_channels,
            use_bias=True,
            bias_init=initializers.zeros,
            kernel_init=initializers.zeros,
        )
        self.adaLN_modulation = nn.Sequential(
            [
                nn.Dense(features=min(self.dim, 1024), use_bias=True),
                swish,
                nn.Dense(features=2 * self.dim, use_bias=True),
            ]
        )

    def __call__(self, x: Array, c: Optional[Array]):
        if c is not None:
            shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=-1)
            x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTModel(nn.Module):

    in_channels: int = 3
    out_channels: int = 3

    input_size: int = 32
    patch_size: int = 2
    dim: int = 512
    n_layers: int = 5
    n_heads: int = 16
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    class_dropout_prob: float = 0.1
    n_classes: int = 200
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.init_conv_seq = nn.Sequential(
            [
                nn.Conv(
                    features=self.dim // 2,
                    kernel_size=(5, 5),
                    padding="SAME",
                    strides=(1, 1),
                ),
                swish,
                nn.GroupNorm(num_groups=32),
                nn.Conv(
                    features=self.dim // 2,
                    kernel_size=(5, 5),
                    padding="SAME",
                    strides=(1, 1),
                ),
                swish,
                nn.GroupNorm(num_groups=32),
            ]
        )

        self.x_embedder = nn.Dense(
            features=self.dim, kernel_init=initializers.xavier_uniform(), use_bias=True
        )

        self.t_embedder = TimestepEmbedder(min(self.dim, 1024), 256)
        self.y_embedder = LabelEmbedder(
            self.n_classes, min(self.dim, 1024), self.class_dropout_prob
        )

        # NOTE this is not in the original DiT, but
        # helps with smaller image sizes, as the positional
        # encodings take a while to learn
        self.pos_embedding = nn.Embed(
            num_embeddings=self.input_size**2,
            features=self.dim,
            embedding_init=initializers.xavier_uniform(),
        )

        self.layers = [
            TransformerBlock(
                layer_id=i,
                dim=self.dim,
                n_heads=self.n_heads,
                multiple_of=self.multiple_of,
                ffn_dim_multiplier=self.ffn_dim_multiplier,
                norm_eps=self.norm_eps,
                dtype=self.dtype,
            )
            for i in range(self.n_layers)
        ]

        self.final_layer = FinalLayer(
            self.dtype, self.dim, self.patch_size, self.out_channels
        )

        max_pos_encoding = (self.input_size // 2) ** 2
        self.freqs_sin, self.freqs_cos = self.precompute_freqs(
            self.dim // self.n_heads, max_pos_encoding
        )

    def patchify(self, x: Array):
        B, C, H, W = x.shape
        x = x.reshape(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = jnp.transpose(x, (0, 2, 4, 1, 3, 5)).reshape(
            B, (H // self.patch_size) * (W // self.patch_size), -1
        )
        return x

    def unpatchify(self, x: Array):
        """
        Convert patched image back to original image.
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape((x.shape[0], h, w, p, p, c))
        # transpose to (B, H, W, C, p, p)
        x = jnp.einsum("nhwpqc->nchpwq", x)
        # reshape to (B, C, H, W)
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

    @staticmethod
    def precompute_freqs(dim: int, maxlen: int, theta: float = 1e4):
        freqs = 1.0 / (theta ** (jnp.arange(0.0, float(dim), 2.0)[: (dim // 2)] / dim))
        t = jnp.arange(maxlen)
        freqs = jnp.outer(t, freqs)
        freqs_cos = jnp.cos(freqs)
        freqs_sin = jnp.sin(freqs)
        return freqs_sin, freqs_cos  # (maxlen, dim/2), (maxlen, dim/2)

    def __call__(self, x: Array, t: Array, y: Array, rng: PRNGKey, train: bool):
        """
        x: latent or image tensor of shape (B, C, H, W)
        t: timestep tensor of shape (B,)
        y: label tensor of shape (B,)
        """

        # transpose since flax uses NCHW format
        x = x.transpose((0, 2, 3, 1))
        x = self.init_conv_seq(x)
        x = x.transpose((0, 3, 1, 2))

        x = self.patchify(x)
        x = self.x_embedder(x)
        x += self.pos_embedding(jnp.arange(x.shape[1]))

        t = self.t_embedder(t)
        y = self.y_embedder(y, train=train, rng=rng)

        adaln_input = t.astype(x.dtype) + y.astype(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_sin, self.freqs_cos, adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, rng, train):
        half = x[: len(x) // 2]
        combined = jnp.concatenate([half, half], axis=0)
        model_out = self(combined, t, y, rng, train)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concatenate([half_eps, half_eps], axis=0)
        return jnp.concatenate([eps, rest], axis=1)
