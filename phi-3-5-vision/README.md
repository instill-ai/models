# Phi 3.5 Vision

## 📖 Introduction

[Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

| Task Type                                                  | Description                                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [Chat](https://www.instill-ai.dev/docs/model/ai-task#chat) | A task to generate conversational style text output base on single or multi-modality input. |

## 🔄 Compatibility Matrix

To ensure smooth integration, please refer to the compatibility matrix below. It outlines the compatible versions of the model, [`instill-core`](https://github.com/instill-ai/instill-core), and the [`python-sdk`](https://github.com/instill-ai/python-sdk).

| Model Version | Instill-Core Version | Python-SDK Version |
| ------------- | -------------------- | ------------------ |
| v0.1.0        | >v0.39.0-beta        | >0.11.0            |

> **Note:** Always ensure that you are using compatible versions to avoid unexpected issues.

## 🚀 Preparation

Follow [this](../README.md) guide to get your custom model up and running! But before you do that, please read through the following sections to have all the necessary files ready.

#### Install Python SDK

Install the compatible [`python-sdk`](https://github.com/instill-ai/python-sdk) version according to the compatibility matrix:

```bash
pip install instill-sdk=={version}
```

#### Get model weights

To download the fine-tuned model weights, please execute the following command:

```bash
git clone https://huggingface.co/microsoft/Phi-3.5-vision-instruct
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/phi-3-5-vision -g -i '{"prompt": "whats in the pic?", "image-url": "https://artifacts.instill.tech/imgs/bear.jpg"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "...",
  "image-url": "https://..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-12 17:26:28,145.145 INFO     [Instill] Starting model image...
2024-09-12 17:26:38,573.573 INFO     [Instill] Deploying model...
2024-09-12 17:27:02,953.953 INFO     [Instill] Running inference...
2024-09-12 17:27:07,323.323 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1726133227,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'The image shows a brown bear '
                                               'sitting on its hind legs in a '
                                               'grassy field with trees in the '
                                               'background.',
                                    'role': 'assistant'}}]}}]
2024-09-12 17:27:10,834.834 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
