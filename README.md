# Instill AI Models

Welcome to our Model Repository! This repository houses a collection of machine learning models designed to perform various AI tasks. They are all prepared and ready to be seamlessly served on [**Instill Core**](https://www.instill-ai.dev/docs/core/introduction) via our MLOps/LLMOps platform [**Instill Model**](https://www.instill-ai.dev/docs/model/introduction).

## Requirements

- [Instill Core](https://www.instill-ai.dev/docs/core/introduction)
- [Instill Python SDK](https://github.com/instill-ai/python-sdk)

## Supported LLM Serving Runtimes

|                           Runtime                           | AMD64 CPU | ARM64 CPU | AMD64 GPU (CUDA) |
| :---------------------------------------------------------: | :-------: | :-------: | :--------------: |
|        [vLLM](https://github.com/vllm-project/vllm)         |     ✅     |     ✅     |        ✅         |
|        [MLC LLM](https://github.com/mlc-ai/mlc-llm)         |     ✅     |     ✅     |        ✅         |
| [Transformers](https://github.com/huggingface/transformers) |     ✅     |     ✅     |        ✅         |

## Available Models

We have a diverse set of models, each optimized for different AI tasks. Please refer to the table below to gain more insight into a specific model, including its configuration, implementation details, and usage. Feel free to check out the README files in the respective model folders:

| Model Name                                                                 | Task Type             | Description                                                                                                                                                                                                          |
| -------------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [deepseek-r1-distill-qwen-1.5b](./deepseek-r1-distill-qwen-1.5b/README.md) | Chat                  | DeepSeek-R1-Distill-Qwen-1.5B is a compact yet powerful language model distilled from DeepSeek-R1, specifically designed for reasoning tasks.                                                                        |
| [phi-3.5-vision-instruct](./phi-3-5-vision/README.md)                      | Chat                  | Phi-3.5-vision is a lightweight, state-of-the-art open multimodal model.                                                                                                                                             |
| [gte-Qwen2-1.5B-instruct](./gte-Qwen2-1.5B-instruct/README.md)             | Embedding             | gte-Qwen2-1.5B-instruct is the latest model in the gte (General Text Embedding) model family.                                                                                                                        |
| [jina-clip-v1](./jina-clip-v1/README.md)                                   | Embedding             | jina-clip-v1 is a state-of-the-art English multimodal (text-image) embedding model.                                                                                                                                  |
| [llama-2-7b-chat](./llama-2-7b-chat/README.md)                             | Chat                  | llama-2-7b-chat is optimized for dialogue use cases.                                                                                                                                                                 |
| [llama-3-8b-instruct](./llama-3-8b-instruct/README.md)                     | Chat                  | llama-3-8b-instruct is an instruction tuned generative text model.                                                                                                                                                   |
| [llamacode-7b](./llamacode-7b/README.md)                                   | Completion            | llamacode-7b is designed for general code synthesis and understanding.                                                                                                                                               |
| [llava-1-6-13b](./llava-1-6-13b/README.md)                                 | Chat                  | llava-1-6-13b is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data.                                                                                                         |
| [mobilenetv2](./mobilenetv2/README.md)                                     | Classification        | mobilenetv2 is a lightweight 53-layer deep CNN model with a smaller number of parameters and an input size of 224×224.                                                                                               |
| [stable-diffusion-xl](./stable-diffusion-xl/README.md)                     | Text to Image         | stable-diffusion-xl is a a latent diffusion model for text-to-image synthesis.                                                                                                                                       |
| [stella-en-1.5B-v5](./stella-en-1.5B-v5/README.md)                         | Embedding             | stella-en-1.5B-v5 is trained based on Alibaba-NLP/gte-large-en-v1.5 and Alibaba-NLP/gte-Qwen2-1.5B-instruct.                                                                                                         |
| [tinyllama](./tinyllama/README.md)                                         | Chat                  | tinyllama is a chat model finetuned on top of TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T.                                                                                                                   |
| [yolov7](./yolov7/README.md)                                               | Object Detection      | yolov7 is a state-of-the-art real-time object detector.                                                                                                                                                              |
| [yolov7-stomata](./yolov7-stomata/README.md)                               | Instance Segmentation | yolov7-stomata is designed for stomata detection and segmentation.                                                                                                                                                   |
| [zephyr-7b](./zephyr-7b/README.md)                                         | Chat                  | zephyr-7b is a series of language models that are trained to act as helpful assistants.                                                                                                                              |
| [gemma-2-27b-it](./gemma-2-27b-it/README.md)                               | Chat                  | Gemma is a family of lightweight, state-of-the-art open models from Google.                                                                                                                                          |
| [qwen-2-5-72b-instruct](./qwen-2-5-72b-instruct/README.md)                 | Chat                  | Qwen2.5 is the latest series of Qwen large language models.                                                                                                                                                          |
| [qwen-2-5-coder-0.5b-instruct](./qwen-2-5-coder-0.5b-instruct/README.md)   | Chat                  | Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen).                                                                                                         |
| [llama-3-2-3b-instruct](./llama-3-2-3b-instruct/README.md)                 | Chat                  | The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out).                                   |
| [llama-3-2-11b-vision-instruct](./llama-3-2-11b-vision-instruct/README.md) | Chat                  | The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). |
| [llama-3-2-90b-vision-instruct](./llama-3-2-90b-vision-instruct/README.md) | Chat                  | The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). |
| [functionary-medium-v3.2](./functionary-medium-v3.2/README.md)             | Chat                  | Functionary is a language model that can interpret and execute functions/plugins.                                                                                                                                    |
| [yi-vl-6b](./yi-vl-6b/README.md)                                           | Chat                  | Yi Vision Language (Yi-VL) model is the open-source, multimodal version of the Yi Large Language Model (LLM) series, enabling content comprehension, recognition, and multi-round conversations about images.        |
| [stable-diffusion-3.5-large](./stable-diffusion-3.5-large/README.md)       | Text to Image         | Stable Diffusion 3.5 Large is a Multimodal Diffusion Transformer (MMDiT) text-to-image model.                                                                                                                        |
| [qwen-2-vl-72b-instruct](./qwen-2-vl-72b-instruct/README.md)               | Chat                  | Qwen2-VL multimodal model.                                                                                                                                                                                           |
| [llama-3-3-70b-instruct](./llama-3-3-70b-instruct/README.md)               | Chat                  | The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out).                                                                         |
| [qwen-2-5-vl-3b-instruct](./qwen-2-5-vl-3b-instruct/README.md)             | Chat                  | Qwen2.5-VL is the latest addition to the Qwen family, featuring enhanced visual understanding, agentic capabilities, visual localization, and structured output generation.                                          |
| [qwen-2-5-vl-7b-instruct](./qwen-2-5-vl-7b-instruct/README.md)             | Chat                  | Qwen2.5-VL is the latest addition to the Qwen family, featuring enhanced visual understanding, agentic capabilities, visual localization, and structured output generation.                                          |

## Getting Started

We leverage Instill Core to provide a seamless experience for serving models. Follow the steps below to quickly get started:

### 1. Instill Core or Instill Cloud

#### Self-host 🔮 Instill Core

Follow [this section](https://www.instill-ai.dev/docs/quickstart#-instill-core) of our quick start guide to get it up and running with self-hosting **Instill Core** on a local or remote instance.

### 2. Create a model namespace

To create a model namespace, follow the steps on the [Create Namespace](https://www.instill-ai.dev/docs/model/create/namespace) page.

### 3. Prepare your model

Find the model you want to serve and download the desired version folder. Also, make sure to check out the particular model folder README to obtain other necessary files, model weights or perform any additional required steps.

### 4. Build your model

Follow the steps on the [Build Model Image](https://www.instill-ai.dev/docs/model/create/build) page, and remember to install the `python-sdk` version according to the compatibility matrix in each model's README.

### 5. Push and deploy your model

Follow the steps on the [Push Model Image](https://www.instill-ai.dev/docs/model/create/push) page to deploy the model to your choice of **Instill Core** or **Instill Cloud**.

## Implement your own custom model

Follow the steps on the [Prepare Model](https://www.instill-ai.dev/docs/model/create/prepare) page to see how to implement your own custom model that can be served on **Instill Core** and **Instill Cloud**! You can also checkout the [step-by-step tutorial](https://www.instill-ai.dev/blog/model-serving-on-instill-core) which walks you through the process of serving your own custom model on **Instill Core**.

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](https://github.com/instill-ai/instill-core/blob/main/.github/CONTRIBUTING.md) file for more details on how to get started.

## 🛠 Troubleshooting

If you encounter any issues, please check our [Documentation](https://www.instill-ai.dev/docs/model/introduction) or open an [issue](https://github.com/instill-ai/instill-core/issues) on GitHub.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/instill-ai/instill-core/blob/main/LICENSE) file for details.
