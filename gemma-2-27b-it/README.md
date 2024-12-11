# Gemma 2 27b it

## 📖 Introduction

Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

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
git clone https://huggingface.co/google/gemma-2-27b-it
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/gemma2 -g -i '{"prompt": "hows life?"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-11-28 21:15:42,947.947 INFO     [Instill] Starting model image...
2024-11-28 21:15:58,526.526 INFO     [Instill] Deploying model...
2024-11-28 21:23:34,019.019 INFO     [Instill] Running inference...
2024-11-28 21:24:24,381.381 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1732800261,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': "As an AI, I don't experience "
                                               'life in the same way humans do',
                                    'role': 'assistant'}}]}}]
2024-11-28 21:24:45,672.672 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
