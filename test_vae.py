import jax
import jax.numpy as jnp
from jax import Array

from vae.vae_flax import load_pretrained_vae

vae, params = load_pretrained_vae()


@jax.jit
def step(sample: Array):
    return vae.apply(params, sample)


test_latent = jnp.zeros((1, 3, 256, 256))

out = step(test_latent)
