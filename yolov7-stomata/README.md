# Yolov7 Instance Segmentation model for stomata detection

## 📖 Introduction

[yolov7-stomata](https://github.com/heiruwu/StomaVision) is a machine learning model designed for stomata detection and segmentation. This implementation leverages [Instill-Core](https://github.com/instill-ai/instill-core) to enhance performance, adaptability, scalability and availability for your inference needs. Whether you're looking to improve accuracy, reduce latency, or customize the model to fit your specific needs, this implementation provides the flexibility you need.

## 🔄 Compatibility Matrix

To ensure smooth integration, please refer to the compatibility matrix below. It outlines the compatible versions of the model, `instill-core`, and the `python-sdk`.

| Model Version | Instill-Core Version | Python-SDK Version |
|---------------|----------------------|--------------------|
| v0.0.1        | <=v0.39.0-beta       | <=0.11.0, >=0.10.2 |
| v0.1.0        | >v0.39.0-beta        | >0.11.0            |

> **Note:** Always ensure that you are using compatible versions to avoid unexpected issues.

## 🚀 Walkthrough

Follow the steps below to get your custom model up and running!

### Prerequisite
- Install the compatible python-sdk version according to the compatibility matrix
```bash
pip install instill-sdk=={version}
```
- Either
  - Self-host Instill-Core instance locally by following the guide [here](https://github.com/instill-ai/instill-core?tab=readme-ov-file#prerequisites)
  - Register a free account on [Instill-Cloud](https://instill.tech)

### 1. Preparing the Model Files

First, create a folder and get the model files from your choice of version. Then, get the model weight by executing the following command.
```bash
curl -o model.pt https://artifacts.instill.tech/model/yolov7-stomata/model.pt
```
Now your folder should look like this:
```
.
├── __init__.py
├── instill.yaml
├── model.py
├── model.pt
├── models
│   ├── __init__.py
│   ├── common.py
│   ├── experimental.py
│   └── yolo.py
└── utils
    ├── __init__.py
    ├── augmentations.py
    ├── autoanchor.py
    ├── general.py
    ├── metrics.py
    ├── segment
    │   ├── __init__.py
    │   └── general.py
    └── torch_utils.py
```

### 2. Building the Model

Next, build your model by packaging it into the required format for deployment on `instill-core` or `instill cloud`. There are a couple flags to pay attention to:
- -t, --tag: We use tag as version for models on instill-core, you can specify your desired version name with this flag. Defaults to `latest`.
- -a --target-arch: Target system's platform which will be serving this model image.

> [!IMPORTANT]
> The model image to be built is platform specific, please specify the target host system's platform if you are not building and serving the model image on the same system.


```bash
# If you are building for CE version of Instill-Core
instill build admin/{your-model-namespace-on-instill-core} --target-arch {arm64,amd64}

# If you are building for Cloud version of Instill-Core
instill build {your-user-name}/{your-model-namespace-on-instill-cloud} --target-arch amd64
```

#### 2.1 Test the Model Locally(Optional)

Before pushing the model onto any Instill-Core instance, you can test if the model can be successfully triggered locally first, by running the following command:

```bash
instill run admin/yolov7-stomata -i '{"image-url": "https://microscopyofnature.com/sites/default/files/2022-03/Mais-stomata-ZW10.jpg", "type": "image-url"}'
```

### 3. Pushing the Model to Instill Core

Deploying your model is a breeze! Push your built model to `instill-core` with the following command:

```bash
# If you are pushing to CE version of Instill-Core
docker login localhost:8080
# username: admin
# password: {your-api-token}
instill push admin/{your-model-namespace-on-instill-core} -u localhost:8080

# If you are pushing to Instill Cloud
docker login api.instill.tech
# username: {your-user-name}
# password: {your-api-token}
instill push {your-user-name}/{your-model-namespace-on-instill-cloud} -u api.instill.tech
```

### 4. Deploying the Model

Once the model is pushed, `Instill-Core` will automatically start provision necessary resources and deploy the model, go to url `{your-user-name}/models/{your-model-namespace}/versions` to check the deployment status.

Congratulations! Your custom model is now live and ready to take on the world. 🌟

## 🛠 Troubleshooting

If you encounter any issues, please check our [Documentation](https://www.instill.tech/docs/model/introduction) or open an [issue](https://github.com/instill-ai/instill-core/issues) on GitHub.

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](https://github.com/instill-ai/instill-core/blob/main/.github/CONTRIBUTING.md) file for more details on how to get started.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/instill-ai/instill-core/blob/main/LICENSE) file for details.

---

Happy Modeling! 💡
