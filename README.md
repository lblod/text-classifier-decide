# Codelist Mapping with Hugging Face Transformers

Basic classification implementation (training) repository for codelist mapping based on huggingface transformers.

## Installation
Clone this repository and install the required packages:
```bash
uv sync
```
## Usage
Make sure you are authenticated with Hugging Face, and obtain a JSON file from Label Studio. Then run the training script with the path to your JSON file and the Hugging Face model ID as arguments:
```commandline
python train.py data.json hf-model-id
```