# MMDiT in Jax

_Inference with a trained MMDiT model on the Oxford Flowers dataset._

![](resources/flowers_result.gif)

This is a reproduction of the [Diffusion Transformer](https://arxiv.org/abs/2212.09748) architecture, along with the [Rectified Flow](https://arxiv.org/abs/2209.03003) sampling method, in Jax. 

### Features

- Training configs for a number of popular reference datasets - CIFAR-10, ImageNet, MNIST, etc. It should be trivial to add additional datasets from HuggingFace.
- DDP support for multi-GPU training (FSDP planned in the near future)
- Mixed precision training, checkpointing, profiler, and TensorBoard logging.
- No dependencies on PyTorch, Tensorflow, etc - just JAX and Flax.

### Results

I was able to achieve comparable validation loss with MNIST and CIFAR-10 to other DiT implementations, so I'm fairly confident in this implementation. I plan to train on ImageNet soon, and will update this README with the results.


![](resources/mnist_result.gif)

_Inference on MNIST_


_Inference on the CIFAR-10 dataset._

![](resources/cifar10_result.gif)


### Why was this project written?

I wanted to brush up on my Jax knowledge, and also hadn't implemented a full MMDiT from scratch before. So I figured I'd try to do both at once! :)

### Setup and Usage

```bash
pip install -r requirements.txt
python train.py
```

`train.py` accepts the following CLI args:
```
n_epochs: Number of epochs to train for.
batch_size: Batch size for training.
learning_rate: Learning rate for the optimizer.
eval_save_steps: Number of steps between evaluation runs and checkpoint saves.
n_eval_batches: Number of batches to evaluate on.
sample_every_n: Number of epochs between sampling runs.
dataset_name: Name of the dataset config to select, valid options are in DATASET_CONFIGS.
profile: Run a single train and eval step, and print out the cost analysis, then exit.
```

TensorBoard is used for logging. Samples will be logged to the `samples` directory, with the X dimension representing batch and Y dimension representing each iteration of the sampling loop.

### Codebase structure

`model.py` contains the DiT implementation. I made the following changes to the [original implementation](https://github.com/facebookresearch/DiT):
- Use a GroupNorm layer in the `PatchEmbed` block.
- Replace `SiLU` with `swish` activation (which are functionally equivalent).
- I added an absolute position embedding - this seems to signficiantly improve training stability with smaller latent / image sizes. Can be easily disabled.

`train.py` contains the training loop, and the main entrypoint for the project. Call the `main()` function to run the training loop; additionally, the `Trainer` class can be used to load and train / inference the model independently of this loop.

I decided to not use either Tensorflow or PyTorch data loading, to keep external dependencies to a minimum. Instead datasets are loaded using [Datasets](https://huggingface.co/docs/datasets/en/index), and processed with the `process_batch` function. To add a new dataset, simply add a new entry to the `DATASET_CONFIGS` dictionary in `train.py`.

### Acknowledgements

I heavily relied on Simo Ryu's [minRF](https://github.com/cloneofsimo/minRF) repo for the original implementation, especially the sampling code for rectified flow; as well as the [original DiT repo](https://github.com/facebookresearch/DiT).

I also learned a lot from Rafi Witten's great [High Performance LLMs in Jax](https://github.com/rwitten/HighPerfLLMs2024) course, especially multi-device training in Jax.


### Todo

- Use [Grain](https://github.com/google/grain) or similar for faster data loading with multiple workers.
- Implement a VAE to allow training on larger image sizes.
- Add support for FSDP, and fp8 training.
- Stable Diffusion 3 support, when the model is released.

_Disclaimer: I work at Stability AI, but this project is not affiliated with the company in any way, and the implementation here is fairly different from SD3's architecture._
