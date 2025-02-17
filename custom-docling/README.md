# Custom model for Docling document parsing

## 📖 Introduction

This is a custom model using `TASK_CUSTOM` to parse documents.

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

To obtain the model weights run the following commands:

```bash
pip install docling
docling-tools models download
cp -r $HOME/.cache/docling/models $HOME/path/to/model/version/docling-models
```

where `$HOME/path/to/model/version/docling-models` is in the same directory as the `model.py` and `instill.yaml` files.

## Test model image

After you've built the model image, and before pushing the model onto any **Instill Core** instance, you can test if the model can be successfully run locally first, by running the following command:

```bash
instill run admin/docling -t v0.1.0 -g -i '{"pdf_content": <INSERT BASE64 STRING>}'
```

For convenience, we have included a sample JSON payload containing a base64 encoded PDF file.
```bash
instill run admin/docling -t v0.1.0 -g -i "$(cat sample_payload.json)"
```

The input payload should strictly follow the the below format

```json
{
  "pdf_content": "<BASE64 STRING>"
}
```

A successful response will return a similar output to that shown below.

```bash
2025-02-17 22:18:17,111.111 INFO     [Instill] Starting model image...
2025-02-17 22:18:22,474.474 INFO     [Instill] Deploying model...
2025-02-17 22:18:30,052.052 INFO     [Instill] Running inference...
2025-02-17 14:19:03,139.139 INFO     [Instill] Outputs:
[{'extracted_images': [],
  'markdown_pages': ['## INVOICE\n'
                     '\n'
                     '#1024\n'
                     '\n'
                     'PAY TO:\n'
                     '\n'
                     '1 2 3   A n y where St., Any City\n'
                     '\n'
                     '1 2 3 - 4 5 6 - 7 8 9 0\n'
                     '\n'
                     'Avery Davis\n'
                     '\n'
                     'Really Great Bank J ohn Smith 000-000\n'
                     '\n'
                     'Bank\n'
                     '\n'
                     'Account Name\n'
                     '\n'
                     'BSB\n'
                     '\n'
                     '0000 0000\n'
                     '\n'
                     'Account Number\n'
                     '\n'
                     'BILLED TO:\n'
                     '\n'
                     'Really Great Company\n'
                     '\n'
                     '| DESCRIPTION            | RATE    | HOURS   | AMOUNT    '
                     '|\n'
                     '|------------------------|---------|---------|-----------|\n'
                     '| Content Plan           | $50/hr  | 4       | $200.00   '
                     '|\n'
                     '| Copy Writing           | $50/hr  | 2       | $100.00   '
                     '|\n'
                     '| Website Design         | $50/hr  | 5       | $250.00   '
                     '|\n'
                     '| Website Development    | $100/hr | 5       | $500.00   '
                     '|\n'
                     '| SEO                    | $50/hr  | 4       | $200.00   '
                     '|\n'
                     '| Sub-Total              |         |         | $1,250.00 '
                     '|\n'
                     '| Package Discount (30%) |         |         | $375.00   '
                     '|\n'
                     '\n'
                     '## TOTAL\n'
                     '\n'
                     '$875.00\n'
                     '\n'
                     'Payment is required within 14 business days of invoice '
                     'date. Please send remittance to '
                     'hello@reallygreatsite.com.\n'
                     '\n'
                     'Thank you for your business.'],
  'page_images': ['data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAC ... ']}]
2025-02-17 22:30:23,039.039 INFO     [Instill] Done
```

Here is the list of flags supported by `instill run` command

- -t, --tag: tag for the model image, default to `latest`
- -g, --gpu: to pass through GPU from host into container or not, depends on if `gpu` is enabled in the config.
- -i, --input: input in json format

---

Happy Modeling! 💡