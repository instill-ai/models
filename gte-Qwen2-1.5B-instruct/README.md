# gte Qwen2 1.5B Instruct

## 📖 Introduction

[gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) is the latest model in the gte (General Text Embedding) model family. The model is built on Qwen2-1.5B LLM model and use the same training data and strategies as the gte-Qwen2-7B-instruct model.

| Task Type                                                            | Description                                                                                                          |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [Embedding](https://www.instill-ai.dev/docs/model/ai-task#embedding) | A task to generate means of representing objects like text, images and audio as points in a continuous vector space. |

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
git clone https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/gte-qwen2-1.5b-instruct -g -i '{"prompt": "hi"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-11 02:36:18,416.416 INFO     [Instill] Starting model image...
2024-09-11 02:36:29,444.444 INFO     [Instill] Deploying model...
2024-09-11 02:37:10,118.118 INFO     [Instill] Running inference...
2024-09-11 02:37:11,585.585 INFO     [Instill] Outputs:
[{'data': {'embeddings': [{'created': 1725993431,
                           'index': 0,
                           'vector': [0.014460443519055843,
                                      0.0885428711771965,
                                      0.02166132815182209,
                                      ...
                                      ...
                                      ...
                                      -0.02432815358042717]}]}}]
2024-09-11 02:39:58,651.651 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
