# About

This application provides an easy way to analyze videos for fall events using Vision Language Models by providing a simple interface. The default model is Qwen2.5-7B-VL - a VLM produced by Qwen - but can be changed to any OpenAI GPT, Anthropic Claude, Google Gemini model that supports video inferece. 

Authors:
1. Maveron Aguares
2. Giovanni Icasiano
3. Prateek Mishra
4. Halima Tebbi

---
# Installation

Follow the steps below to set up the environment and install all dependencies.

<br>

## Install Python 3.13.9

Ensure Python 3.13.9 is installed on your system.

Check the version:

```bash
python --version
```

or: 


```bash
python3 --version
```

<br>

## Create a Virtual Environment

Inside the project directory, create a virtual environment:

```bash
python -m venv venv
```

Activate it: 

**Windows**

```bash
venv\Scripts\activate
```

**Linux/MacOS**

```bash
source venv/bin/activate
```

<br>

### Install Dependencies

With the virtual environment activated, install the required packages:


```bash
pip install --upgrade pip
pip install -r requirements.txt
```
<br>

### Run the application

```bash
python app.py
```
---

# User Guide

## Setting environment variables

A `.env.example` file is provided to show how to format a `.env` file for the application. OpenAI, Anthropic, and Gemini keys are required to run their respective models.

## Setting configuration

These settings come from `config.yaml` and control how each model behaves inside the application.

### Qwen 2.5 VL Configuration

| Setting            | Example Value                               | Description |
|--------------------|----------------------------------------------|-------------|
| `model`            | `"Qwen/Qwen2.5-VL-7B-Instruct"`             | The HuggingFace model ID to load. |
| `download_local`   | `True`                                       | If `True`, downloads the model locally; if `False`, loads it from HuggingFace online. |
| `local_folder`     | `"./models/qwen2.5vl"`                       | Directory where the Qwen model will be stored when downloaded locally. |
| `max_new_tokens`   | `128`                                        | Maximum number of tokens Qwen is allowed to generate. |
| `temperature`      | `0`                                          | Controls randomness (0 = deterministic). |

---

### OpenAI (GPT-4o / GPT-4 Vision) Configuration

| Setting | Example Value     | Description |
|---------|--------------------|-------------|
| `name`  | `"GPT-4 Vision"`   | Display name used in the application. |
| `model` | `"gpt-4o"`         | The OpenAI model ID used for inference. |

---

### Google Gemini Configuration

| Setting | Example Value         | Description |
|---------|------------------------|-------------|
| `name`  | `"Gemini 1.5 Flash"`   | Display name for Gemini. |
| `model` | `"gemini-1.5-flash"`   | The Google model ID used for native video analysis. |

---

### Anthropic Claude Configuration

| Setting | Example Value                         | Description |
|---------|----------------------------------------|-------------|
| `name`  | `"Claude 3.5 Sonnet"`                  | Display name for Claude. |
| `model` | `"claude-3-5-sonnet-20241022"`         | Anthropic model ID used for inference. |
---
