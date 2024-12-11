# Qwen2.5 Coder 32B Instruct GGUF

## 📖 Introduction

Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen).

| Task Type                                                | Description                                                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [Chat](https://www.instill.tech/docs/model/ai-task#chat) | A task to generate conversational style text output base on single or multi-modality input. |

## 🔄 Compatibility Matrix

To ensure smooth integration, please refer to the compatibility matrix below. It outlines the compatible versions of the model, [`instill-core`](https://github.com/instill-ai/instill-core), and the [`python-sdk`](https://github.com/instill-ai/python-sdk).

| Model Version | Instill-Core Version | Python-SDK Version |
| ------------- | -------------------- | ------------------ |
| v0.1.0        | >v0.46.0-beta        | >0.16.0            |

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
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct-GGUF --include "qwen2.5-coder-32b-instruct-q4_0.gguf" --local-dir ./
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/qwen-2-5-coder-32b-instruct -g -i '{"prompt": "describe python in one line"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-11-29 06:47:40,285.285 INFO     [Instill] Starting model image...
2024-11-29 06:47:55,895.895 INFO     [Instill] Deploying model...
2024-11-29 06:51:25,876.876 INFO     [Instill] Running inference...
2024-11-28 06:51:28,139.139 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1732834288,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'Python is a high-level, '
                                               'interpreted programming '
                                               'language known for its '
                                               'readability and simplicity.',
                                    'role': 'assistant'}}]}}]
2024-11-29 06:51:33,384.384 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
