# Llama 3.2 11B Vision

## 📖 Introduction

The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out).

| Task Type                                                  | Description                                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [Chat](https://www.instill-ai.dev/docs/model/ai-task#chat) | A task to generate conversational style text output base on single or multi-modality input. |

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
git clone git@hf.co:meta-llama/Llama-3.2-11B-Vision-Instruct
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/llama-3-2-11b-vision-instruct -g -i '{"prompt": "whats in the pic?", "image-url": "https://artifacts.instill.tech/imgs/bear.jpg"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-12-02 05:56:42,585.585 INFO     [Instill] Starting model image...
2024-12-02 05:56:58,202.202 INFO     [Instill] Deploying model...
2024-12-02 05:57:25,122.122 INFO     [Instill] Running inference...
2024-12-01 05:57:32,328.328 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1733090252,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'The image depicts a brown bear '
                                               'sitting upright on its hind '
                                               'legs, with its front paws '
                                               'raised in the air. The bear is '
                                               'positioned centrally in the '
                                               'frame, facing the camera '
                                               'directly. Its fur appears to '
                                               'be a light brown color, and it '
                                               'has',
                                    'role': 'assistant'}}]}}]
2024-12-02 05:57:36,100.100 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
