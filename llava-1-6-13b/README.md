# Llava 1.6 13B

## 📖 Introduction

[LLaVA](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture.

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
git clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/llava-1-6-13b -g -i '{"prompt": "whats in the pic?", "image-url": "https://artifacts.instill.tech/imgs/bear.jpg"}'
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
2024-09-11 01:54:17,944.944 INFO     [Instill] Starting model image...
2024-09-11 01:54:28,926.926 INFO     [Instill] Deploying model...
2024-09-11 01:57:46,629.629 INFO     [Instill] Running inference...
2024-09-11 01:58:15,578.578 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1725991095,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'The image shows a bear sitting '
                                               'in a field. The bear appears '
                                               'to be in a relaxed posture '
                                               'with one paw raised in the air '
                                               'as if waving. The background '
                                               'suggests a natural, open '
                                               'habitat, possibly a grassy '
                                               'area with trees',
                                    'role': 'assistant'}}]}}]
2024-09-11 01:58:19,447.447 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
