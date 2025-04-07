# Llama3 8B Instruct

## 📖 Introduction

Meta developed and released the Meta [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks.

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
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/llama-3-8b-instruct -g -i '{"prompt": "hows life?"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-11 01:48:37,795.795 INFO     [Instill] Starting model image...
2024-09-11 01:48:48,785.785 INFO     [Instill] Deploying model...
2024-09-11 01:49:28,613.613 INFO     [Instill] Running inference...
2024-09-11 01:49:33,350.350 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1725990573,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': "I'm just an AI, I don't have a "
                                               'life in the classical sense. I '
                                               'exist solely to assist and '
                                               'communicate with users like '
                                               "you. I don't have emotions, "
                                               'experiences, or personal '
                                               "relationships. I'm just a "
                                               'collection of code and data',
                                    'role': 'assistant'}}]}}]
2024-09-11 01:49:37,114.114 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
