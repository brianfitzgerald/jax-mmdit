import jax
from jax import Array, random
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict, TypedDict
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from utils import denormalize_images

ln = False


class DDPMScheduleValues(TypedDict):
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_cumprod: jnp.ndarray
    sqrt_alpha_cumprod: jnp.ndarray
    sqrt_one_minus_alphacumprod: jnp.ndarray
    sqrt_recip_alphas: jnp.ndarray
    sqrt_recipm1_alphas: jnp.ndarray


def ddpm_schedules(beta_start: float, beta_end: float, n_T: int) -> DDPMScheduleValues:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    This computes the values for each variable required in the sampling process, for all timesteps.
    """
    assert beta_start < beta_end < 1.0, "beta1 and beta2 must be in (0, 1)"

    # beta at each timestep, this is the variance schedule - i.e. how much variance to use for adding noise at each timestep
    betas = jnp.linspace(beta_start, beta_end, n_T)
    # this is the amount of data to keep at each timestep, i.e 1-data
    alphas = 1.0 - betas
    # cumulative product of alphas
    alpha_cumprod = jnp.cumprod(alphas)

    # compute sqrt of alpha_cumprod
    sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod)
    sqrt_one_minus_alphacumprod = jnp.sqrt(1.0 - alpha_cumprod)
    sqrt_recip_alphas = jnp.sqrt(1.0 / alphas)
    sqrt_recipm1_alphas = jnp.sqrt(1.0 / alphas - 1)

    return DDPMScheduleValues(
        betas=betas,
        alphas=alphas,
        alpha_cumprod=alpha_cumprod,
        sqrt_alpha_cumprod=sqrt_alpha_cumprod,
        sqrt_one_minus_alphacumprod=sqrt_one_minus_alphacumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        sqrt_recipm1_alphas=sqrt_recipm1_alphas,
    )


def rectified_flow_step(
    state: TrainState,
    image: Array,
    label: Array,
    prng_key: Array,
    training: bool = True,
) -> Tuple[Array, Optional[TrainState]]:
    prng_key, step_key = random.split(prng_key)

    def rectified_flow_loss(params: FrozenDict, x: Array, cond: Array, rng_key: Array):
        """
        Rectified Flow sampling.
        https://huggingface.co/blog/Isamu136/insta-rectified-flow
        Sample a single timestep and get loss
        """
        b = x.shape[0]
        if ln:
            t = jax.nn.sigmoid(random.normal(rng_key, (b,)))
        else:
            t = random.uniform(rng_key, (b,))

        texp = t.reshape([b] + [1] * (len(x.shape) - 1))  # expand dims
        z1 = random.normal(rng_key, x.shape)  # random noise
        # zt is the mixture of x and z1 at timestep t
        zt = (1 - texp) * x + texp * z1
        # vtheta is the output of the model - predicts velocity at timestep t
        vtheta = state.apply_fn({"params": params}, zt, t, cond, rng_key, training)

        # MSE of the model output and the noise
        # can think of this as predicting velocity for each timestep, with the expected
        # velocity being constant for all timesteps
        batchwise_mse = jnp.mean(
            (z1 - x - vtheta) ** 2, axis=tuple(range(1, len(x.shape)))
        )

        return batchwise_mse.mean()

    if training:
        loss, grads = jax.value_and_grad(rectified_flow_loss)(
            state.params, image, label, step_key
        )
        return loss, state.apply_gradients(grads=grads)
    else:
        loss = rectified_flow_loss(state.params, image, label, step_key)
        return loss, state


def ddpm_step(
    state: TrainState,
    image: Array,
    label: Array,
    prng_key: Array,
    noise_schedule: Dict[str, jnp.ndarray],
    n_timesteps: int,
    training: bool,
) -> Tuple[Array, Optional[TrainState]]:
    prng_key, step_key = random.split(prng_key)

    def ddpm_loss(params: FrozenDict, x: Array, cond: Array, rng_key: Array):
        """
        # Sample a
        """

        _ts = random.randint(rng_key, (x.shape[0],), 1, n_timesteps + 1)
        eps = random.normal(rng_key, x.shape)

        # this gives the noise level for the timestep, which the model is conditioned on
        x_t = noise_schedule["sqrtab"][_ts] * x + noise_schedule["sqrtmab"][_ts] * eps

        ts_scaled = _ts / n_timesteps
        loss = state.apply_fn({"params": params}, x_t, ts_scaled, cond, rng_key)
        return loss

    if training:
        loss, grads = jax.value_and_grad(ddpm_loss)(
            state.params, image, label, step_key
        )
        return loss, state.apply_gradients(grads=grads)
    else:
        loss = ddpm_loss(state.params, image, label, step_key)
        return loss, None


@jax.jit
def sample_loop(
    z, t, cond, train_state: TrainState, cfg, null_cond, dt, rng_key
):
    v_cond = train_state.apply_fn({"params": train_state.params}, z, t, cond, rng_key, False)
    # CFG
    if null_cond is not None:
        v_uncond = train_state.apply_fn(
            {"params": train_state.params}, z, t, null_cond, rng_key, False
        )
        v_cond = v_uncond + cfg * (v_cond - v_uncond)

    z = z - dt * v_cond
    return z


# TODO jit this fn
def rectified_flow_sample(
    train_state: TrainState,
    z: Array,
    cond: Array,
    rng_key: Array,
    null_cond=None,
    sample_steps: int = 30,
    cfg: float = 2.0,
) -> List[List[Image.Image]]:
    b = z.shape[0]
    dt = 1.0 / sample_steps
    # reshape to (b, 1, 1, 1, 1) for broadcasting
    dt = jnp.array([dt] * b).reshape([b] + [1] * (len(z.shape) - 1))
    outs = [z]

    for i in tqdm(range(sample_steps, 0, -1), desc="Sampling", leave=False):
        t = i / sample_steps
        t = jnp.array([t] * b)
        outs.append(z)
        z = sample_loop(
            z, t, cond, train_state, cfg, null_cond, dt, rng_key
        )

    images = jnp.stack(outs, axis=0)
    images = images.transpose((0, 1, 3, 4, 2))
    images = denormalize_images(images)
    img_mode = "L" if images.shape[-1] == 1 else "RGB"
    if img_mode == "L":
        images = images[..., 0]
    images_cpu = images.__array__()
    batches = []
    for step in range(sample_steps):
        images_batch = [Image.fromarray(img, mode=img_mode) for img in images_cpu[step]]
        batches.append(images_batch)
    return batches
