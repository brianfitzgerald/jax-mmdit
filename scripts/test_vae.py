import jax
import jax.numpy as jnp
from jax import Array

from vae.vae_flax import load_pretrained_vae

vae, params = load_pretrained_vae("stabilityai/sdxl-vae")


@jax.jit
def step(sample: Array):
    return vae.apply(params, sample)


sample_size = vae.config.sample_size
test_latent = jnp.zeros((1, 3, sample_size, sample_size))

out = step(test_latent)
