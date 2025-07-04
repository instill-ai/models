# Llama 3.2 3B Instruct

## 📖 Introduction

The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out).

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
git clone git@hf.co:meta-llama/Llama-3.2-3B-Instruct
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/llama-3-2-3b-instruct -g -i '{"prompt": "how much do you know about python? summarize in one line"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-12-02 02:13:19,927.927 INFO     [Instill] Starting model image...
2024-12-02 02:13:30,491.491 INFO     [Instill] Deploying model...
2024-12-02 02:13:51,266.266 INFO     [Instill] Running inference...
2024-12-01 02:13:56,909.909 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1733076836,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'I have knowledge of Python, a '
                                               'high-level, interpreted '
                                               'programming language known for '
                                               'its simplicity, readability, '
                                               'and versatility, with '
                                               'applications in web '
                                               'development, data analysis, '
                                               'machine learning, automation, '
                                               'and more.',
                                    'role': 'assistant'}}]}}]
2024-12-02 02:14:00,463.463 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
