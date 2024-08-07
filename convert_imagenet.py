from streaming.base.format.mds.encodings import Encoding, _encodings
from datasets import Dataset, concatenate_datasets
from streaming import StreamingDataset
from tqdm import tqdm
import numpy as np
from typing import Any
from loguru import logger
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from loguru import logger

from vae.vae_flax import load_pretrained_vae


dataset: Dataset = Dataset.from_dict({"label": [], "vae_output": []})
dataset.set_format(type="numpy")


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return x


_encodings["uint8"] = uint8


remote_train_dir = "./vae_mds"
local_train_dir = "./local_train_dir"

train_dataset = StreamingDataset(
    local=local_train_dir,
    remote="evanarlian/imagenet_1k_resized_256",
    split=None,
    shuffle=True,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size=32,
)

save_every = train_dataset.length // 4

logger.info(
    f"Train dataset length: {train_dataset.length}, saving every {save_every} samples"
)

new_rows = []

vae, params = load_pretrained_vae("pcuenq/sd-vae-ft-mse-flax", True)
sample_size = vae.config.sample_size


@jax.jit
def vae_encode(sample: Array) -> Array:
    return vae.apply(params, sample, method="decode")  # type: ignore


for i, sample in enumerate(tqdm(train_dataset, dynamic_ncols=True)):
    try:
        sample["vae_output"] = vae_encode(
            jnp.array(sample["image"]).reshape(1, 4, 32, 32)
        ).astype(np.int8)
        sample["label"] = np.array(sample["label"]).astype(np.int8)
        new_rows.append(sample)
    except Exception as e:
        logger.error(f"Error at iteration {i}: {e}")
    if i % save_every == 0 and i > 0:
        logger.info(f"Uploading at iteration {i}...")
        dataset_new_rows = Dataset.from_list(new_rows)
        concat_dataset = concatenate_datasets([dataset, dataset_new_rows])
        concat_dataset.push_to_hub("roborovski/imagenet-int8-flax")
