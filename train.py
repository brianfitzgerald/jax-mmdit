import functools
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import fire
import jax
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as jnp
import optax
import orbax
import orbax.checkpoint
import orbax.checkpoint as ocp
from chex import PRNGKey
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from flax import linen as nn
from flax.linen.summary import tabulate
from flax.training.train_state import TrainState
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.stages import Compiled, Wrapped
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm

from labels import IMAGENET_LABELS_NAMES
from model import DiTModel
from sampling import rectified_flow_sample, rectified_flow_step
from utils import center_crop, ensure_directory, image_grid, normalize_images
from profiling import trace_module_calls

jax.experimental.compilation_cache.compilation_cache.set_cache_dir("jit_cache")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def format_float_for_display(val: float | int) -> str:
    return f"{val:06.3f}"


@dataclass
class ModelConfig:
    """
    Modifiable params for the model's architecture.
    """

    dim: int
    n_layers: int
    n_heads: int
    patch_size: int = 2


@dataclass
class DatasetConfig:
    """
    Contains all the necessary information to load a dataset and preprocess it.
    """

    hf_dataset_uri: str
    n_classes: int
    latent_size: int
    eval_split_name: str = "test"
    train_split_name: str = "train"
    image_field_name: str = "image"
    label_field_name: str = "label"
    label_names: Optional[List[str]] = None
    n_channels: int = 3
    n_labels_to_sample: Optional[int] = None
    batch_size: int = 256

    model_config: ModelConfig = ModelConfig(dim=64, n_layers=6, n_heads=4)


DATASET_CONFIGS = {
    # https://huggingface.co/datasets/zh-plus/tiny-imagenet
    "tiny_imagenet": DatasetConfig(
        hf_dataset_uri="zh-plus/tiny-imagenet",
        n_classes=200,
        eval_split_name="valid",
        latent_size=64,
        label_names=list(IMAGENET_LABELS_NAMES.values()),
        n_labels_to_sample=10,
        batch_size=64,
        model_config=ModelConfig(dim=1152, n_layers=28, n_heads=16, patch_size=2),
    ),
    # https://huggingface.co/datasets/cifar10
    "cifar10": DatasetConfig(
        hf_dataset_uri="cifar10",
        n_classes=10,
        image_field_name="img",
        latent_size=32,
        label_names=[
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        model_config=ModelConfig(dim=256, n_layers=10, n_heads=8),
    ),
    # TODO find the class counts and resize with preprocessor
    "butterflies": DatasetConfig(
        hf_dataset_uri="ceyda/smithsonian_butterflies",
        n_channels=3,
        n_classes=25,
        latent_size=64,
    ),
    "mnist": DatasetConfig(
        hf_dataset_uri="mnist",
        n_channels=1,
        n_classes=10,
        latent_size=28,
    ),
    "flowers": DatasetConfig(
        hf_dataset_uri="nelorth/oxford-flowers",
        n_channels=3,
        n_classes=102,
        latent_size=32,
        batch_size=128,
        n_labels_to_sample=16,
        model_config=ModelConfig(dim=64, n_layers=10, n_heads=8),
    ),
    "fashion_mnist": DatasetConfig(
        hf_dataset_uri="fashion_mnist",
        n_channels=1,
        n_classes=10,
        latent_size=28,
        label_names=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    ),
}


class Trainer:

    def __init__(
        self,
        rng: PRNGKey,
        dataset_config: DatasetConfig,
        learning_rate: float = 5e-5,
        profile: bool = False,
        half_precision: bool = False,
    ) -> None:
        self.optimizer = optax.chain(
            optax.adam(learning_rate=learning_rate),
        )
        init_key, self.train_key = random.split(rng, 2)
        latent_size, n_channels = dataset_config.latent_size, dataset_config.n_channels
        dtype = jnp.float16 if half_precision else jnp.float32

        self.model = DiTModel(
            dim=dataset_config.model_config.dim,
            n_layers=dataset_config.model_config.n_layers,
            n_heads=dataset_config.model_config.n_heads,
            input_size=latent_size,
            in_channels=n_channels,
            out_channels=n_channels,
            n_classes=dataset_config.n_classes,
            dtype=dtype,
        )
        print(self.model)
        n_devices = len(jax.devices())

        # x, y, t
        input_values = (
            jnp.ones((n_devices, n_channels, latent_size, latent_size)),
            jnp.ones((n_devices)),
            jnp.ones((n_devices), dtype=jnp.int32),
        )

        def create_train_state(x, y, t, model, optimizer):
            variables = model.init(
                init_key,
                x=x,
                t=t,
                y=y,
                rng=init_key,
                train=True,
            )
            train_state = TrainState.create(
                apply_fn=self.model.apply, params=variables["params"], tx=optimizer
            )
            return train_state

        logging.info(f"Available devices: {jax.devices()}")

        # Create a device mesh according to the physical layout of the devices.
        # device_mesh is just an ndarray
        device_mesh = mesh_utils.create_device_mesh((n_devices, 1))
        logging.info(f"Device mesh: {device_mesh}")

        # Async checkpointer for saving checkpoints across processes
        base_dir_abs = os.getcwd()
        self.checkpoint_manager = ocp.CheckpointManager(f"{base_dir_abs}/checkpoints")

        # The axes are (data, model), so the mesh is (n_devices, 1) as the model is replicated across devices.
        # This object corresponds the axis names to the layout of the physical devices,
        # so that sharding a tensor along the axes shards according to the corresponding device_mesh layout.
        # i.e. with device layout of (8, 1), data would be replicated to all devices, and model would be replicated to 1 device.
        self.mesh = Mesh(device_mesh, axis_names=("data", "model"))
        logging.info(f"Mesh: {self.mesh}")

        def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
            """
            Get a NamedSharding for a given PartitionSpec, and the device mesh.
            A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
            """
            return NamedSharding(self.mesh, pspec)

        # This shards the first dimension of the input data (batch dim) across the data axis of the mesh.
        x_sharding = get_sharding_for_spec(PartitionSpec("data"))

        # Returns a pytree of shapes for the train state
        train_state_sharding_shape = jax.eval_shape(
            functools.partial(
                create_train_state, model=self.model, optimizer=self.optimizer
            ),
            *input_values,
        )

        # Get the PartitionSpec for all the variables in the train state
        train_state_sharding = nn.get_sharding(train_state_sharding_shape, self.mesh)
        input_sharding: Any = (x_sharding, x_sharding, x_sharding)

        logging.info(f"Initializing model...")
        # Shard the train_state so so that it's replicated across devices
        jit_create_train_state_fn = jax.jit(
            create_train_state,
            static_argnums=(3, 4),
            in_shardings=input_sharding,  # type: ignore
            out_shardings=train_state_sharding,
        )
        self.train_state = jit_create_train_state_fn(
            *input_values, self.model, self.optimizer
        )
        parameter_count = sum(
            x.size for x in jax.tree_util.tree_leaves(self.train_state.params)
        )
        logging.info(f"Model parameter count: {parameter_count}")

        if profile:
            logging.info("Running model inspection...")
            tabulate_fn = tabulate(
                self.model,
                self.train_key,
                compute_flops=True,
                compute_vjp_flops=True,
            )
            calls = trace_module_calls(self.model, *input_values)

        logging.info("JIT compiling step functions...")

        step_in_sharding: Any = (
            train_state_sharding,
            x_sharding,
            x_sharding,
            x_sharding,
        )
        step_out_sharding: Any = (
            get_sharding_for_spec(PartitionSpec()),
            train_state_sharding,
        )
        self.train_step: Wrapped = jax.jit(
            functools.partial(rectified_flow_step, training=True),
            in_shardings=step_in_sharding,
            out_shardings=step_out_sharding,
        )

        self.eval_step: Wrapped = jax.jit(
            functools.partial(rectified_flow_step, training=False),
            in_shardings=step_in_sharding,
            out_shardings=step_out_sharding,
        )

        if profile:
            logging.info("AOT compiling step functions...")
            compiled_step: Compiled = self.train_step.lower(
                self.train_state, *input_values[:2], init_key
            ).compile()
            train_cost_analysis = compiled_step.cost_analysis()
            logging.info(f"Steps compiled, train cost analysis: {train_cost_analysis}")

    def save_checkpoint(self, global_step: int):
        self.checkpoint_manager.save(
            global_step, self.train_state, args=ocp.args.StandardSave(self.train_state)
        )


def process_batch(
    batch: dict,
    latent_size: int,
    n_channels: int,
    label_field_name: str,
    image_field_name: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process a batch of samples from the dataset.
    Provide the entire batch to the train/eval step, and the in_sharding will partition across
    devices.
    If an image is not square, it will be center cropped to the smaller dimension, before being resized to the latent size.
    """

    images: List[Image.Image] = batch[image_field_name]
    img_mode = "L" if n_channels == 1 else "RGB"
    for i, image in enumerate(images):
        if image.width != image.height:
            smaller_dim = min(image.width, image.height)
            image = center_crop(image, smaller_dim, smaller_dim)
        images[i] = image.resize((latent_size, latent_size)).convert(img_mode)
    image_jnp = jnp.asarray(images, dtype=jnp.float32)
    if n_channels == 1:
        image_jnp = image_jnp[:, :, :, None]
    # convert to NCHW format
    image_jnp = image_jnp.transpose((0, 3, 1, 2))
    image_jnp = normalize_images(image_jnp)
    label = jnp.asarray(batch[label_field_name], dtype=jnp.float32)
    return image_jnp, label


def run_eval(
    eval_dataset: Dataset,
    n_eval_batches: int,
    dataset_config: DatasetConfig,
    trainer: Trainer,
    rng: PRNGKey,
    summary_writer: SummaryWriter,
    iter_description_dict: dict,
    global_step: int,
    do_sample: bool,
    epoch: int,
):
    """
    Run evaluation on the eval subset, and optionally sample the model
    """
    num_eval_batches = len(eval_dataset) // dataset_config.batch_size
    eval_iter = tqdm(
        eval_dataset.iter(batch_size=16, drop_last_batch=True),
        leave=False,
        total=num_eval_batches,
    )
    for j, eval_batch in enumerate(eval_iter):
        if j >= n_eval_batches:
            break

        # Eval loss
        images, labels = process_batch(
            eval_batch,
            dataset_config.latent_size,
            dataset_config.n_channels,
            dataset_config.label_field_name,
            dataset_config.image_field_name,
        )
        eval_loss, trainer.train_state = trainer.eval_step(
            trainer.train_state, images, labels, rng
        )
        iter_description_dict.update({"eval_loss": format_float_for_display(eval_loss)})
        eval_iter.set_postfix(iter_description_dict)
        summary_writer.add_scalar("eval_loss", eval_loss, global_step)

        # Sampling
        if do_sample:
            sample_key, rng = random.split(rng)
            n_labels_to_sample = (
                dataset_config.n_labels_to_sample
                if dataset_config.n_labels_to_sample
                else dataset_config.n_classes
            )
            noise_shape = (
                n_labels_to_sample,
                dataset_config.n_channels,
                dataset_config.latent_size,
                dataset_config.latent_size,
            )
            init_noise = random.normal(rng, noise_shape)
            labels = jnp.arange(0, n_labels_to_sample)
            null_cond = jnp.ones_like(labels) * 10
            samples = rectified_flow_sample(
                trainer.train_state,
                init_noise,
                labels,
                sample_key,
                null_cond=null_cond,
                sample_steps=50,
            )
            grid = image_grid(samples)
            sample_img_filename = f"samples/epoch_{epoch}_globalstep_{global_step}.png"
            grid.save(sample_img_filename)


def main(
    n_epochs: int = 100,
    learning_rate: float = 5e-4,
    eval_save_steps: int = 250,
    n_eval_batches: int = 1,
    sample_every_n: int = 1,
    dataset_name: str = "mnist",
    profile: bool = False,
    half_precision: bool = False,
    **kwargs,
):
    """
    Arguments:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        eval_save_steps: Number of steps between evaluation runs and checkpoint saves.
        n_eval_batches: Number of batches to evaluate on.
        sample_every_n: Number of epochs between sampling runs.
        dataset_name: Name of the dataset config to select, valid options are in DATASET_CONFIGS.
        profile: Run a single train and eval step, and print out the cost analysis, then exit.
        half_precision: case the model to fp16 for training.
    """

    assert not kwargs, f"Unrecognized arguments: {kwargs.keys()}"
    assert dataset_name in DATASET_CONFIGS, f"Invalid dataset name: {dataset_name}"

    dataset_config = DATASET_CONFIGS[dataset_name]
    dataset = cast(DatasetDict, load_dataset(dataset_config.hf_dataset_uri))
    train_dataset = dataset[dataset_config.train_split_name]
    eval_dataset = dataset[dataset_config.eval_split_name]

    rng = random.PRNGKey(0)

    trainer = Trainer(rng, dataset_config, learning_rate, profile, half_precision)

    profiler_trace_dir = "traces"

    summary_writer = SummaryWriter(flush_secs=1, max_queue=1)
    ensure_directory("samples", clear=True)

    iter_description_dict = {"loss": 0.0, "eval_loss": 0.0, "epoch": 0, "step": 0}

    n_evals = 0
    for epoch in range(n_epochs):
        iter_description_dict.update({"epoch": epoch})
        n_batches = len(train_dataset) // dataset_config.batch_size
        train_iter = tqdm(
            train_dataset.iter(
                batch_size=dataset_config.batch_size, drop_last_batch=True
            ),
            total=n_batches,
            leave=False,
        )
        for i, batch in enumerate(train_iter):

            global_step = epoch * (len(train_dataset) // dataset_config.batch_size) + i

            # Train step
            images, labels = process_batch(
                batch,
                dataset_config.latent_size,
                dataset_config.n_channels,
                dataset_config.label_field_name,
                dataset_config.image_field_name,
            )
            step_key = random.PRNGKey(global_step)

            if profile:
                # profile_ctx = jax.profiler.trace(
                #     profiler_trace_dir, create_perfetto_link=True
                # )
                profile_ctx = nullcontext()
            else:
                profile_ctx = nullcontext()

            with profile_ctx:
                step_start_time = time.perf_counter()
                train_loss, updated_state = trainer.train_step(
                    trainer.train_state, images, labels, step_key
                )
                trainer.train_state = updated_state
                summary_writer.add_scalar(
                    "train_step_time",
                    time.perf_counter() - step_start_time,
                    global_step,
                )

            iter_description_dict.update(
                {
                    "loss": format_float_for_display(train_loss),
                    "epoch": epoch,
                    "step": i,
                }
            )
            summary_writer.add_scalar("train_loss", train_loss, global_step)

            train_iter.set_postfix(iter_description_dict)

            if i % eval_save_steps == 0 or profile:
                trainer.save_checkpoint(global_step)
                run_eval(
                    eval_dataset,
                    n_eval_batches,
                    dataset_config,
                    trainer,
                    rng,
                    summary_writer,
                    iter_description_dict,
                    global_step,
                    n_evals % sample_every_n == 0,
                    epoch,
                )

            if profile:
                break

        if epoch % sample_every_n == 0 and not profile:
            trainer.save_checkpoint(global_step)
            run_eval(
                eval_dataset,
                n_eval_batches,
                dataset_config,
                trainer,
                rng,
                summary_writer,
                iter_description_dict,
                global_step,
                True,
                epoch,
            )


if __name__ == "__main__":
    fire.Fire(main)
