# Yolov7 Instance Segmentation model for stomata detection

## 📖 Introduction

[yolov7-stomata](https://github.com/YaoChengLab/StomaVision) is a machine learning model designed for stomata detection and segmentation, check out the link to lean more.

| Task Type                                                                                  | Description                                                                                         |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| [Instance Segmentation](https://www.instill.tech/docs/model/ai-task#instance-segmentation) | A vision task to detect and delineate multiple objects of pre-defined categories in an input image. |

## 🔄 Compatibility Matrix

To ensure smooth integration, please refer to the compatibility matrix below. It outlines the compatible versions of the model, [`instill-core`](https://github.com/instill-ai/instill-core), and the [`python-sdk`](https://github.com/instill-ai/python-sdk).

| Model Version | Instill-Core Version | Python-SDK Version |
| ------------- | -------------------- | ------------------ |
| v0.0.1        | <=v0.39.0-beta       | <=0.11.0, >=0.10.2 |
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
curl -o model.pt https://artifacts.instill.tech/model/yolov7-stomata/model.pt
```

## Test model image

After you've built the model image and before pushing the model onto any Instill-Core instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/yolov7-stomata -i '{"image-url": "https://microscopyofnature.com/sites/default/files/2022-03/Mais-stomata-ZW10.jpg", "type": "image-url"}'
```

And you should get a output like this

```bash
2024-09-04 00:17:41,512.512 INFO     [Instill] Starting model image...
2024-09-04 00:17:51,820.820 INFO     [Instill] Deploying model...
2024-09-04 00:17:53,510.510 INFO     [Instill] Running inference...
2024-09-03 09:17:57,564.564 INFO     [Instill] Outputs:
[
    {
        "data": {
            "objects": [
                {
                    "bounding-box": {
                        "height": 40,
                        "left": 306,
                        "top": 502,
                        "width": 67
                    },
                    "category": "outer_line",
                    "rle": "137,13,23,19,19,22,17,24,15,25,14,27,13,27,13,27,12,29,11,29,11,30,10,31,9,31,9,32,8,33,7,15,3,15,7,15,4,14,7,14,6,13,7,14,6,13,7,14,5,14,7,15,4,14,7,15,4,14,7,16,2,15,7,16,2,15,7,33,7,33,7,33,7,33,7,33,7,33,7,33,7,10,1,22,7,9,3,21,7,8,4,21,7,8,4,21,7,7,6,20,7,7,6,20,7,7,6,20,7,7,6,20,7,7,6,20,7,7,6,19,8,7,6,19,8,7,6,18,9,8,5,17,10,8,4,17,11,8,4,17,11,9,2,17,12,28,12,28,12,27,13,27,13,27,13,26,14,25,15,24,16,23,17,22,18,21,19,20,20,20,21,18,23,15,27,10,59",
                    "score": 0.9110509753227234
                },
                {
                    "bounding-box": {
                        "height": 35,
                        "left": 93,
                        "top": 26,
                        "width": 61
                    },
                    "category": "outer_line",
                    "rle": "117,13,20,18,15,21,14,22,12,23,11,25,9,27,7,29,5,31,3,32,3,33,2,33,2,33,2,33,2,15,3,15,2,14,5,14,2,14,6,13,2,14,6,13,2,13,7,13,2,13,7,13,2,13,7,13,2,13,7,13,2,13,7,13,2,13,7,13,2,13,7,13,2,14,6,13,2,14,6,13,2,14,6,13,2,14,5,14,2,14,5,14,2,14,5,14,2,15,3,15,2,16,2,15,2,33,2,33,2,33,2,33,2,33,2,33,2,33,2,33,2,32,3,31,4,31,4,30,5,29,6,29,6,28,8,26,9,26,10,24,12,23,13,21,15,19,17,17,20,12,25,9,49",
                    "score": 0.8888306021690369
                },
                {
                    "bounding-box": {
                        "height": 31,
                        "left": 261,
                        "top": 336,
                        "width": 59
                    },
                    "category": "outer_line",
                    "rle": "11,10,19,17,13,19,11,20,10,22,9,22,8,23,6,25,5,26,5,26,4,27,4,28,3,28,3,28,3,28,3,15,4,9,3,14,6,8,3,14,6,8,3,14,6,8,3,14,6,9,2,14,6,9,2,14,6,9,2,14,6,9,2,14,6,9,2,14,6,9,2,15,4,10,2,29,2,29,2,29,2,29,2,28,3,28,3,28,3,28,3,28,3,28,3,28,3,28,3,28,3,28,3,28,3,27,4,27,4,27,4,27,4,27,4,26,5,25,6,23,8,22,9,21,10,21,11,19,13,17,16,13,21,7,78",
                    "score": 0.8836774826049805
                }
                ...
                ...
                ...
            ]
        }
    }
]
2024-09-04 00:18:00,141.141 INFO     [Instill] Done
```

---

Happy Modeling! 💡
