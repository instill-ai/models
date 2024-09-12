# Yolov7 object detector

## 📖 Introduction

[yolov7](https://github.com/WongKinYiu/yolov7) is a machine learning model trained for object detection task, check out the [paper](https://arxiv.org/abs/2207.02696) to lean more.

| Task Type                                                                        | Description                                                                             |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [Object Detection](https://www.instill.tech/docs/model/ai-task#object-detection) | A vision task to localise multiple objects of pre-defined categories in an input image. |

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
curl -o model.onnx https://artifacts.instill.tech/model/yolov7/model.onnx
```

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/yolov7 -i '{"image-url": "https://artifacts.instill.tech/imgs/bear.jpg", "type": "image-url"}'
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
2024-09-10 16:12:20,040.040 INFO     [Instill] Starting model image...
2024-09-10 16:12:30,257.257 INFO     [Instill] Deploying model...
2024-09-10 16:12:31,827.827 INFO     [Instill] Running inference...
2024-09-10 16:12:33,703.703 INFO     [Instill] Outputs:
[
    {'data': {'objects': [
                {'bounding-box': {'height': 757.0,
                                         'left': 290.0,
                                         'top': 84.0,
                                         'width': 554.0
                    },
                        'category': 'bear',
                        'score': 0.9658154845237732
                }
            ]
        }
    }
]
2024-09-10 16:12:37,387.387 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡
