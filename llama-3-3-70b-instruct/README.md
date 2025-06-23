# Llama 3.3 70B Instruct

## 📖 Introduction

The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.

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
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF --include "Llama-3.3-70B-Instruct-IQ4_XS.gguf" --local-dir ./
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/llama-3-3-70b-instruct -g -ng 2 -i '{"prompt": "how much do you know about python? summarize in one line"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-12-10 06:07:44,089.089 INFO     [Instill] Starting model image...
2024-12-10 06:08:00,010.010 INFO     [Instill] Deploying model...
2024-12-10 06:13:07,892.892 INFO     [Instill] Running inference...
2024-12-09 06:13:18,671.671 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1733782398,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'I have extensive knowledge of '
                                               'Python, including its syntax, '
                                               'data structures, file '
                                               'operations, object-oriented '
                                               'programming, popular libraries '
                                               '(e.g., NumPy, pandas, Flask, '
                                               'Django), and various '
                                               'applications (e.g., web '
                                               'development, data science, '
                                               'machine',
                                    'role': 'assistant'}}]}}]
2024-12-10 06:13:25,810.810 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
