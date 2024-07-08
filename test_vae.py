import jax
import jax.numpy as jnp
from jax import Array
from datasets import load_dataset

from vae.vae_flax import load_pretrained_vae

vae, params = load_pretrained_vae("stabilityai/sdxl-vae")
sample_size = vae.config.sample_size

@jax.jit
def step(sample: Array):
    return vae.apply(params, sample, method="decode")

sample = jnp.zeros((1, 3, sample_size, sample_size))


dataset = load_dataset("roborovski/imagenet-int8-flax")

first_sample = next(iter(dataset["train"])) # type: ignore
sample_tensor = jnp.array(first_sample["vae_output"]).reshape(1, 4, 32, 32)
out = step(sample_tensor)
