# Phi 3.5 Vision

## 📖 Introduction

[Yi Vision Language](https://huggingface.co/01-ai/Yi-VL-6B) (Yi-VL) model is the open-source, multimodal version of the Yi Large Language Model (LLM) series, enabling content comprehension, recognition, and multi-round conversations about images.
Yi-VL demonstrates exceptional performance, ranking first among all existing open-source models in the latest benchmarks including MMMU in English and CMMMU in Chinese (based on data available up to January 2024).

| Task Type                                                  | Description                                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [Chat](https://www.instill-ai.dev/docs/model/ai-task#chat) | A task to generate conversational style text output base on single or multi-modality input. |

## 🔄 Compatibility Matrix

To ensure smooth integration, please refer to the compatibility matrix below. It outlines the compatible versions of the model, [`instill-core`](https://github.com/instill-ai/instill-core), and the [`python-sdk`](https://github.com/instill-ai/python-sdk).

| Instill Core Version | Python SDK Version |
| -------------------- | ------------------ |
| >= v0.51.0           | >= v0.18.0         |

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
git clone https://huggingface.co/01-ai/Yi-VL-6B
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/yivl-6b -g -i '{"prompt": "whats in the pic?", "image-url": "https://artifacts.instill.tech/imgs/bear.jpg"}'
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
2024-12-03 03:11:00,407.407 INFO     [Instill] Starting model image...
2024-12-03 03:11:15,989.989 INFO     [Instill] Deploying model...
2024-12-03 03:11:42,386.386 INFO     [Instill] Running inference...
2024-12-02 03:11:47,130.130 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1733166707,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'a bear in a field',
                                    'role': 'assistant'}}]}}]
2024-12-03 03:11:50,829.829 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
