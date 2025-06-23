# Qwen2.5 VL 7B Instruct

## 📖 Introduction

The latest addition to the Qwen family: Qwen2.5-VL.

Key Enhancements:
- Understand things visually: Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images.

- Being agentic: Qwen2.5-VL directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use.

- Capable of visual localization in different formats: Qwen2.5-VL can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes.

- Generating structured outputs: for data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.

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
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/qwen-2-5-vl-7b-instruct -g -i '{"prompt": "whats in the pic? describe in one sentence", "image-url": "https://artifacts.instill.tech/imgs/bear.jpg"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2025-02-04 00:25:58,180.180 INFO     [Instill] Starting model image...
2025-02-04 00:26:08,583.583 INFO     [Instill] Deploying model...
2025-02-04 00:26:13,019.019 INFO     [Instill] Running inference...
2025-02-03 16:26:17,050.050 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1738628777,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': 'The image shows a large brown '
                                               'bear sitting on its hind legs '
                                               'in a grassy field, with one '
                                               'paw raised as if waving.',
                                    'role': 'assistant'}}]}}]
2025-02-04 00:26:20,516.516 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡