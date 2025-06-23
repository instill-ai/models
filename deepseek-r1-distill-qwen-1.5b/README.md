# DeepSeek R1 Distill Qwen 1.5B

## 📖 Introduction

[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) is a compact yet powerful language model distilled from DeepSeek-R1, specifically designed for reasoning tasks. This model is based on the Qwen2.5-Math-1.5B architecture and has been fine-tuned using knowledge distilled from the larger DeepSeek-R1 model.

Key features:
- Built on Qwen2.5-Math-1.5B as the base model
- Distilled from DeepSeek-R1 using reasoning-focused training data
- Optimized for mathematical reasoning, coding, and general problem-solving tasks
- Efficient reasoning capabilities in a compact 1.5B parameter model
- Inherits DeepSeek-R1's advanced reasoning patterns through distillation
- Compatible with standard transformer-based architectures
- Optimized for deployment in resource-constrained environments

| Task Type                                                  | Description                                         |
| ---------------------------------------------------------- | --------------------------------------------------- |
| [Chat](https://www.instill-ai.dev/docs/model/ai-task#chat) | A task to generate conversational style text output |

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
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/deepseek-r1-distill-qwen-1.5b -g -i '{"prompt": "what is the capital of England?"}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2025-01-31 15:10:25,002.002 INFO     [Instill] Starting model image...
2025-01-31 15:10:30,245.245 INFO     [Instill] Deploying model...
2025-01-31 15:10:38,919.919 INFO     [Instill] Running inference...
2025-01-31 07:10:56,594.594 INFO     [Instill] Outputs:
[{'data': {'choices': [{'created': 1738336256,
                        'finish-reason': 'length',
                        'index': 0,
                        'message': {'content': '<｜begin▁of▁sentence｜><｜User｜>what '
                                               'is the capital of '
                                               'England?<｜Assistant｜><think>\n'
                                               '\n'
                                               '</think>\n'
                                               '\n'
                                               'The capital of England is '
                                               'London.',
                                    'role': 'assistant'}}]}}]
2025-01-31 15:11:00,325.325 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
