# Custom model

## 📖 Introduction

This is a custom model to showcase `TASK_CUSTOM`

| Task Type                                                    | Description                                   |
| ------------------------------------------------------------ | --------------------------------------------- |
| [Custom](https://www.instill.tech/docs/model/ai-task#custom) | A custom task for arbitrary input and output. |

## 🔄 Compatibility Matrix

To ensure smooth integration, please refer to the compatibility matrix below. It outlines the compatible versions of the model, [`instill-core`](https://github.com/instill-ai/instill-core), and the [`python-sdk`](https://github.com/instill-ai/python-sdk).

| Model Version | Instill-Core Version | Python-SDK Version |
| ------------- | -------------------- | ------------------ |
| v0.1.0        | >v0.39.0-beta        | >0.16.1            |

> **Note:** Always ensure that you are using compatible versions to avoid unexpected issues.

## 🚀 Preparation

Follow [this](../README.md) guide to get your custom model up and running! But before you do that, please read through the following sections to have all the necessary files ready.

#### Install Python SDK

Install the compatible [`python-sdk`](https://github.com/instill-ai/python-sdk) version according to the compatibility matrix:

```bash
pip install instill-sdk=={version}
```

#### Get model weights

no model weight needed for this model.

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/custom -i '{"key1": "val1", "key2": "val2"}'
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
2025-02-17 18:09:25,457.457 INFO     [Instill] Starting model image...
2025-02-17 18:09:30,781.781 INFO     [Instill] Deploying model...
2025-02-17 18:09:34,137.137 INFO     [Instill] Running inference...
2025-02-17 18:09:34,894.894 INFO     [Instill] Outputs:
[{'key1': 'val1', 'key2': 'val2'}]
2025-02-17 18:09:38,320.320 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
