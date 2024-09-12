# Jina CLIP V1

## 📖 Introduction

[jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1) is a state-of-the-art English multimodal (text-image) embedding model.

| Task Type                                                          | Description                                                                                                          |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| [Embedding](https://www.instill.tech/docs/model/ai-task#embedding) | A task to generate means of representing objects like text, images and audio as points in a continuous vector space. |

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
git clone https://huggingface.co/jinaai/jina-clip-v1
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/jina-clip-v1 -g -i '{"text" : "hi", "image": "https://artifacts.instill.tech/imgs/bear.jpg"}'
```

The input payload should strictly follow the the below format

```json
{
  "text": "...",
  "image": "https://",
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-11 02:42:38,605.605 INFO     [Instill] Starting model image...
2024-09-11 02:42:49,440.440 INFO     [Instill] Deploying model...
2024-09-11 02:43:16,851.851 INFO     [Instill] Running inference...
2024-09-11 02:43:19,756.756 INFO     [Instill] Outputs:
[{'data': {'embeddings': [{'created': 1725993799,
                           'index': 0,
                           'vector': [-0.042002271860837936,
                                      0.002093376824632287,
                                      0.007119686808437109,
                                      ...
                                      ...
                                      ...
                                      -0.0350787378847599]},
                          {'created': 1725993799,
                           'index': 1,
                           'vector': [-0.07706715911626816,
                                      -0.006987405009567738,
                                      0.0100631695240736,
                                      ...
                                      ...
                                      ...
                                      -0.02432815358042717]}]}}]
2024-09-11 02:43:23,487.487 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
