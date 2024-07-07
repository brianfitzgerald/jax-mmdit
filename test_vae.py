from typing import Dict
import jax
import jax.numpy as jnp

from vae.vae_flax import FlaxAutoencoderKL

vae = FlaxAutoencoderKL()

@jax.jit
def step(sample: Dict):
    return vae(sample['latent'])


test_latent = jnp.zeros((1, 32, 32, 4))

out = step({"latent": test_latent})
