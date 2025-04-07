# Stable Diffusion 3.5 Large

## 📖 Introduction

[Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.

| Task Type                                                                    | Description                                                                                                                                                                           |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Text to Image](https://www.instill-ai.dev/docs/model/ai-task#text-to-image) | A task to generate images from text inputs. Generally, the task takes descriptive text prompts as the input, and outputs generated images in Base64 format based on the text prompts. |

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
git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-large
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/sd35large -g -i '{"prompt": "cat napping"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-11 02:23:19,079.079 INFO     [Instill] Starting model image...
2024-09-11 02:23:29,503.503 INFO     [Instill] Deploying model...
2024-09-11 02:25:28,298.298 INFO     [Instill] Running inference...
2024-09-11 02:25:45,534.534 INFO     [Instill] Outputs:
[{'data': {'choices': [{'finish-reason': 'success',
                        'image': 'data:image/jpeg;base64,/9j/7gAOQWRvY...'}]}}]
2024-09-11 02:25:46,962.962 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
