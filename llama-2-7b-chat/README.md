# Llama2 7B Chat

## 📖 Introduction

[Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This is the repository for the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.

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
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/llama-2-7b-chat -g -i '{"prompt": "hi"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-11 01:44:12,423.423 INFO     [Instill] Starting model image...
2024-09-11 01:44:22,843.843 INFO     [Instill] Deploying model...
2024-09-11 01:44:52,935.935 INFO     [Instill] Running inference...
2024-09-11 01:44:56,534.534 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1725990296,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': "Hello! It's nice to meet you. "
                                               'Is there something I can help '
                                               'you with or would you like to '
                                               'chat?',
                                    'role': 'assistant'}}]}}]
2024-09-11 01:45:00,240.240 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
