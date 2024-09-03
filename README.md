# Instill AI Models

Welcome to our Model Repository! This repository houses a collection of machine learning models designed to perform various AI tasks, and be seamlessly served on [Instill Core](https://github.com/instill-ai/instill-core) or [Instill Cloud](www.instill.tech).

## Available Models

We have a diverse set of models, each optimized for different AI tasks. Please refer to the table below to gain more insight into a specific model, including its configuration, implementation details, and usage. Feel free to check out the README files in the respective model folders:

| Model Name                                                     | Task Type             | Description                                                                                                            |
| -------------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [gte-Qwen2-1.5B-instruct](./gte-Qwen2-1.5B-instruct/README.md) | Embedding             | gte-Qwen2-1.5B-instruct is the latest model in the gte (General Text Embedding) model family.                          |
| [jina-clip-v1](./jina-clip-v1/README.md)                       | Embedding             | jina-clip-v1 is a state-of-the-art English multimodal (text-image) embedding model.                                    |
| [llama2-7b-chat](./llama2-7b-chat/README.md)                   | Chat                  | llama2-7b-chat is optimized for dialogue use cases.                                                                    |
| [llama3-8b-instruct](./llama3-8b-instruct/README.md)           | Chat                  | llama3-8b-instruct is an instruction tuned generative text model.                                                      |
| [llamacode-7b](./llamacode-7b/README.md)                       | Completion            | llamacode-7b is designed for general code synthesis and understanding.                                                 |
| [llava-1-6-13b](./llava-1-6-13b/README.md)                     | Chat                  | llava-1-6-13b is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data.           |
| [mobilenetv2](./mobilenetv2/README.md)                         | Classification        | mobilenetv2 is a lightweight 53-layer deep CNN model with a smaller number of parameters and an input size of 224×224. |
| [stable-diffusion-xl](./stable-diffusion-xl/README.md)         | Text to Image         | stable-diffusion-xl is a a latent diffusion model for text-to-image synthesis.                                         |
| [stella-en-1.5B-v5](./stella-en-1.5B-v5/README.md)             | Embedding             | stella-en-1.5B-v5 is trained based on Alibaba-NLP/gte-large-en-v1.5 and Alibaba-NLP/gte-Qwen2-1.5B-instruct.           |
| [tinyllama](./tinyllama/README.md)                             | Chat                  | tinyllama is a chat model finetuned on top of TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T.                     |
| [yolov7](./yolov7/README.md)                                   | Object Detection      | yolov7 is a state-of-the-art real-time object detector.                                                                |
| [yolov7-stomata](./yolov7-stomata/README.md)                   | Instance Segmentation | yolov7-stomata is designed for stomata detection and segmentation.                                                     |
| [zephyr-7b](./zephyr-7b/README.md)                             | Chat                  | zephyr-7b is a series of language models that are trained to act as helpful.assistants.                                |

## Getting Started

We leverage Instill Core to provide a seamless experience for serving models. Follow the steps below to quickly get started:

### 1. Instill Core or Instill Cloud

#### Self-host Instill-Core

- Follow this [quickstart guide](https://www.instill.tech/docs/quickstart) to get it up and running.

#### Instill Cloud

- Go to our [website](https://www.instill.tech/) and check out what we have to offer.

### 2. Create a Namespace

To create a namespace, follow the steps in [Create a Namespace Documentation](https://www.instill.tech/docs/model/create/namespace).

### 3. Prepare your model

Find the model you want to serve and download the desired version folder, also check out the particular model folder README to obtain other necessary files and model weights or perform necessary steps.

### 4. Build your model

Follow the steps in [Build Your Model Documentation](https://www.instill.tech/docs/model/create/build), and remember to install the python-sdk version according to the compatibility matrix in each model's README.

### 5. Push and deploy your model

Follow the steps in [Push Your Model Documentation](https://www.instill.tech/docs/model/create/push) to deploy the model.

## Implement your own custom model

Follow the steps in [Prepare Model](https://www.instill.tech/docs/model/create/prepare) to see how to implement your own custom model that can be served on Instill Core and Instill Cloud!

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](https://github.com/instill-ai/instill-core/blob/main/.github/CONTRIBUTING.md) file for more details on how to get started.

## 🛠 Troubleshooting

If you encounter any issues, please check our [Documentation](https://www.instill.tech/docs/model/introduction) or open an [issue](https://github.com/instill-ai/instill-core/issues) on GitHub.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/instill-ai/instill-core/blob/main/LICENSE) file for details.
