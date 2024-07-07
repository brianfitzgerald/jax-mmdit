from typing import Dict
import jax
import jax.numpy as jnp

from vae.vae_flax import FlaxAutoencoderKL


vae = FlaxAutoencoderKL()


@jax.jit
def step(sample: Dict, vae: FlaxAutoencoderKL):
    return vae.apply(sample)


test_latent = jnp.zeros((1, 32))

out = step({"latent": test_latent}, vae)
