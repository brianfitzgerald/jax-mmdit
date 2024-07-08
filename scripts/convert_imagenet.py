from streaming.base.format.mds.encodings import Encoding, _encodings
from datasets import load_dataset, Dataset, concatenate_datasets
from streaming import StreamingDataset
import pyarrow.parquet as pq
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Any
from loguru import logger

dataset: Dataset = Dataset.from_dict({"label": [], "vae_output": []})
dataset.set_format(type="numpy")


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8


remote_train_dir = "./vae_mds"
local_train_dir = "./local_train_dir"

train_dataset = StreamingDataset(
    local=local_train_dir,
    remote=remote_train_dir,
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

for i, sample in enumerate(tqdm(train_dataset)):
    sample["vae_output"] = sample["vae_output"].astype(np.int32)
    sample['label'] = np.array(sample['label']).astype(np.int64)
    new_rows.append(sample)
    if i % save_every == 0 and i > 0:
        logger.info(f"Uploading at iteration {i}...")
        dataset_new_rows = Dataset.from_list(new_rows)
        concat_dataset = concatenate_datasets([dataset, dataset_new_rows])
        concat_dataset.push_to_hub("roborovski/imagenet-int8-flax")
