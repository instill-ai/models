# Llama 3.2 3B Instruct

## 📖 Introduction

Functionary is a language model that can interpret and execute functions/plugins.

The model determines when to execute functions, whether in parallel or serially, and can understand their outputs. It only triggers functions as needed. Function definitions are given as JSON Schema Objects, similar to OpenAI GPT function calls.

> **Note:** Function calling capability will be added soon!

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
huggingface-cli download bartowski/functionary-medium-v3.2-GGUF --include "functionary-medium-v3.2-Q6_K_L.gguf" --local-dir ./
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/functionary -g -i '{"prompt": "how much do you know about python? summarize in one line"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-12-02 18:08:18,913.913 INFO     [Instill] Starting model image...
2024-12-02 18:08:39,918.918 INFO     [Instill] Deploying model...
2024-12-02 18:12:12,779.779 INFO     [Instill] Running inference...
2024-12-02 18:12:15,652.652 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1733134335,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'I know enough about Python to '
                                               'write clean, efficient, and '
                                               'readable code that solves',
                                    'role': 'assistant'}}]}}]
2024-12-02 18:12:19,418.418 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
