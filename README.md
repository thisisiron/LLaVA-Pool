# Llava Pool
<p align="center">
    <img src="assets/llavapool.png" width=128>
</p>

This project provides functionalities for training and configuring Vision-Language Models (VLM).

## Features
- Open Vision Language Model: Ex. Qwen2-VL, Pixtral, LLama 3.2 Vision
- Training methods for VLMs: Pre-Training, Supervised Fine-Tuning

## Install
To install LLaVA-Pool, follow these commands in order. The flash-attn package is required for GPU acceleration and improved performance.
```
git clone https://github.com/thisisiron/LLaVA-Pool.git
cd LLaVA-Pool
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Data Preparation
Provide detailed instructions on how to prepare the data for training.

## Pre-training Model
Magma(Multimodal AI Generation and Model Architecture) is a pre-trained model that can be used for various tasks. It is designed to be flexible and adaptable to different use cases. You select the model you want to use based on your needs. If you want to use the Qwen2.5 as the LLM model and SigLIP as the vision model, modify the config of the Magma model to use them.

## SFT Model List

| Model | Converter |
| --- | --- |
| Qwen2-VL | qwen2_vl |
| Qwen2.5-VL | qwen2_vl |
| Llama 3.2 Vision | llama3.2_vision |
| Pixtral | pixtral |
| InternVL2.5 | internvl2_5 |

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## References
This repository was built based on LLaMA-Factory.

- LLaMA-Factory
- LLaVA-NeXT
