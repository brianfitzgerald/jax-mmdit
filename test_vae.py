import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from datasets import load_dataset
from PIL import Image
from loguru import logger
jax.experimental.compilation_cache.compilation_cache.set_cache_dir("jit_cache")

from vae.vae_flax import load_pretrained_vae

import jax.experimental.compilation_cache.compilation_cache

vae, params = load_pretrained_vae("pcuenq/sd-vae-ft-mse-flax", True)
sample_size = vae.config.sample_size

@jax.jit
def step(sample: Array):
    return vae.apply(params, sample, method="decode")

sample = jnp.zeros((1, 3, sample_size, sample_size))

dataset = load_dataset("roborovski/imagenet-int8-flax")

first_sample = next(iter(dataset["train"])) # type: ignore
sample_tensor = jnp.array(first_sample["vae_output"]).reshape(1, 4, 32, 32)
out = step(sample_tensor)
out_np = np.array(out[0])
img = Image.fromarray((out_np.transpose(1,2,0) * 255).astype("uint8"))
img.save("test.png")
logger.info("Saved image to test.png")