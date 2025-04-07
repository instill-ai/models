# MobilenetV2

## 📖 Introduction

[MobileNetV2](https://arxiv.org/abs/1801.04381) is a new mobile architecture that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes.

| Task Type                                                                            | Description                                                                         |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| [Classification](https://www.instill-ai.dev/docs/model/ai-task#image-classification) | Vision task to assign a single pre-defined category label to an entire input image. |

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
curl -o model.onnx https://artifacts.instill.tech/model/mobilenetv2/model.onnx
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/mobilenetv2 -i '{"image-url": "https://artifacts.instill.tech/imgs/bear.jpg", "type": "image-url"}'
```

The input payload should strictly follow the the below format

```json
{
  "image-url": "https://...",
  "type": "image-url"
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-11 02:19:20,870.870 INFO     [Instill] Starting model image...
2024-09-11 02:19:31,172.172 INFO     [Instill] Deploying model...
2024-09-11 02:19:44,971.971 INFO     [Instill] Running inference...
2024-09-11 02:19:46,726.726 INFO     [Instill] Outputs:
[{'data': {'category': 'brown bear', 'score': 0.9989921450614929}}]
2024-09-11 02:19:50,993.993 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
