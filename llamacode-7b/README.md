# Llama Code 7B

## 📖 Introduction

[Code Llama](https://huggingface.co/codellama/CodeLlama-7b-hf) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 34 billion parameters. This is the repository for the base 7B version in the Hugging Face Transformers format. This model is designed for general code synthesis and understanding.

| Task Type                                                              | Description                                                          |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [Completion](https://www.instill-ai.dev/docs/model/ai-task#completion) | A task to generate natural continuation texts based on input prompt. |

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
git clone https://huggingface.co/codellama/CodeLlama-7b-hf
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run instill-ai/llamacode-7b -g -i '{"prompt": "go func("}'
```

The input payload should strictly follow the the below format

```json
{
  "prompt": "..."
}
```

A successful response will return a similar output to that shown below.

```bash
2024-09-11 02:03:26,813.813 INFO     [Instill] Starting model image...
2024-09-11 02:03:37,228.228 INFO     [Instill] Deploying model...
2024-09-11 02:04:37,750.750 INFO     [Instill] Running inference...
2024-09-11 02:04:42,434.434 INFO     [Instill] Outputs:
[{'data': {'choices': [{'content': 'go func(interface{}) error, interceptor '
                                   'grpc.UnaryServerInterceptor) (interface{}, '
                                   'error) {\n'
                                   '\tin := new(SayHelloReq)\n'
                                   '\tif err := dec(in); err!= nil {',
                        'created': 1725991482,
                        'finish-reason': 'length',
                        'index': 0}]}}]
2024-09-11 02:04:46,147.147 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
