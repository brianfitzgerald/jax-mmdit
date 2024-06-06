# MMDiT in Jax

![](resources/cifar10_result.gif)

_Inference with a trained MMDiT model on CIFAR-10_

This is a reproduction of the [Diffusion Transformer](https://arxiv.org/abs/2212.09748) architecture, along with the [Rectified Flow](https://arxiv.org/abs/2209.03003) sampling method, in Jax. 

### Features

- Training configs for a number of popular reference datasets - CIFAR-10, ImageNet, MNIST, etc. It should be trivial to add additional datasets from HuggingFace.
- DDP support - FSDP coming in the near future.
- Mixed precision training, checkpointing, profiler, and TensorBoard logging.
- No dependencies on PyTorch, Tensorflow, etc - just JAX and Flax.

### Validation

I've trained models on CIFAR-10, and several other reference datasets, and was able to achieve competitive results. I plan on training on ImageNet soon, and will update this section with the results.

![](resources/mnist_result.gif)

_Inference on MNIST_

![](resources/flowers_result.gif)

_Inference on Oxford Flowers dataset_

### Why was this project written?

I wanted to brush up on my Jax knowledge, and also hadn't implemented a full MMDiT from scratch before. So I figured I'd try to do both at once!

### Future features

- Use [Grain](https://github.com/google/grain) or similar for faster data loading with multiple workers.
- Implement a VAE to allow training on larger image sizes.
- Add support for FSDP, and fp8 training.
- Stable Diffusion 3 support, when the model is released. (I work at Stability - but this project is not affiliated with the company in any way, and the implementation here is fairly different from SD3's architecture.)


### Acknowledgements

I heavily relied on Simo Ryu's [minRF](https://github.com/cloneofsimo/minRF) repo for the original implementation, especially the sampling code for rectified flow; as well as the [original DiT repo](https://github.com/facebookresearch/DiT).

I also learned a lot from Rafi Witten's great [High Performance LLMs in Jax](https://github.com/rwitten/HighPerfLLMs2024) course, especially multi-device training in Jax.

### Setup

```bash
pip install -r requirements.txt
python train.py
```

Documentation for CLI args is available in `train.py`'s `main` function.
