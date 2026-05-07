# 🧪 Edge AI Audio Models Testbench & Evaluation Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-blueviolet)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005ced)

## 📑 Table of Contents
- [Abstract](#abstract)
- [Operational Modes & Test Methodology](#operational-modes)
- [Supported Models Library](#supported-models)
- [Tools & Utility Scripts](#tools-and-utilities)
- [Hardware Profiling Benchmarks](#hardware-benchmarks)
- [Results and Performance Discussion](#results-discussion)
- [Getting Started](#getting-started)

---

<a id="abstract"></a>
## 📖 Abstract
This repository provides a highly modular, end-to-end benchmarking framework designed to evaluate and compare locally hosted AI models for Conversational Voice Assistants. 

It allows developers to plug-and-play different **Speech-to-Text (STT)**, **Large Language Models (LLM)**, and **Text-to-Speech (TTS)** engines, rigorously profiling their performance on Edge hardware (e.g., Mini PCs, SBCs) without relying on cloud APIs.

---

<a id="operational-modes"></a>
## ⚙️ Operational Modes & Test Methodology

The core orchestrator (`DefMain.py`) is structured into **5 distinct operational modes**, allowing both interactive usage and automated batch benchmarking. 

Whenever an OpenVINO or hardware-agnostic model is selected, **the framework dynamically prompts the user to select the target inference device (CPU, GPU, or NPU)**, enabling precise hardware-level profiling.

### [1] Full Pipeline (STT -> LLM -> TTS)
An interactive, real-time mode. It captures microphone input, transcribes it, generates an LLM response, and synthesizes the audio output, measuring the total Time-To-First-Audio (TTFA) and pipeline latency.

### [2] STT WER Evaluation
* **Dataset:** LibriSpeech Streaming Dataset.
* **Methodology:** The framework downloads standardized audio samples, passes them through the selected STT engine, and compares the transcription against the ground truth using the `jiwer` library to calculate the **Word Error Rate (WER)**.

### [3] Liquid ONNX Engine Benchmark
A dedicated benchmarking suite designed to profile the execution speed and system impact of continuous/liquid state-space models optimized in ONNX format.

### [4] TTS RTF Evaluation
* **Dataset:** LibriSpeech Streaming Dataset (Text subset).
* **Methodology:** Feeds standardized textual sentences to the selected TTS engine. It calculates the **Real-Time Factor (RTF)** by dividing the time taken to generate the audio by the actual duration of the generated audio (RTF < 1.0 means the model generates audio faster than real-time playback).

### [5] LLM Tk/s & Semantic Similarity Evaluation
* **Dataset:** SQuAD (Stanford Question Answering Dataset).
* **Methodology:** Feeds context paragraphs and questions to the LLM. It measures generation speed in **Tokens per Second (Tk/s)** and uses NLP metrics to evaluate the semantic similarity between the LLM's generated answer and the expected ground truth.

---

<a id="supported-models"></a>
## 🧩 Supported Models Library

The framework currently supports a vast array of cutting-edge models, carefully integrated to run completely offline. 

### 🎤 Speech-to-Text (STT) Engines
* **Whisper Family (OpenVINO):** Tiny OV, Base OV, Small OV, Medium OV, V3 Turbo OV `[CPU, GPU]`
* **Distil-Whisper (OpenVINO):** Small, Large v3.5 `[CPU, GPU]`
* **Moonshine:** Tiny (27M), Base (61M) `[CPU]`
* **Fun-ASR-Nano:** Native `[CPU]` and OpenVINO Optimized `[CPU, GPU]`
* **Qwen3-ASR 0.6B:** OpenVINO Optimized `[CPU, GPU]`
* **Kyutai STT:** Native `[CPU]`

### 🧠 Large Language Models (LLMs)
* **Qwen 2.5 1.5B Instruct:** OpenVINO int4 `[CPU, GPU, NPU]`
* **DeepSeek R1 1.5B Distill:** OpenVINO int4 `[CPU, GPU, NPU]`
* **Qwen OV:** Standard OpenVINO `[CPU, GPU, NPU]`
* **SmolLM2 360M:** Native `[CPU]`
* **Gemma 3 270M:** Native `[CPU]`

### 🔊 Text-to-Speech (TTS) Engines
* **Kokoro:** OpenVINO (`af_heart`) `[CPU, GPU, NPU]` and Native `[CPU]`
* **Piper:** ONNX Runtime `[CPU]`
* **KittenTTS:** Nano and Mini 0.8 (ONNX) `[CPU]`
* **Soprano 1.1 80M:** Native `[CPU]`
* **Pocket-TTS (Kyutai):** Native `[CPU]`
* **Supertonic-2:** ONNX Runtime `[CPU]`
* **OuteTTS 0.1:** 350M LLaMa-based `[CPU]`
* **Qwen3-TTS 0.6B:** OpenVINO Optimized `[CPU, GPU]`
* **VoxCPM 0.5B:** Diffusion-based `[CPU]`

---

<a id="tools-and-utilities"></a>
## 🛠️ Tools & Utility Scripts

This directory contains accessory scripts and utility tools designed to support development, testing, and environment setup.

**Provided Tools:**
* `Single_Model_Metrics.py`: An automated script that checks whether a specific model is eligible for testing on the Modular Pipeline.
* `HF_Model_Scraper.py`: An automated script for interacting with the Hugging Face Hub. It is used to analyze, filter, or download the specific model weights and configuration files required by the local assistant.

---

<a id="hardware-benchmarks"></a>
## 📊 Audio Models Testbench (Hardware Profiling)

This directory contains the scripts used for benchmarking and analyzing the performance of AI models running on the [Khadas Mind 2 AI Maker Kit](https://www.khadas.com/product-page/mind-maker-kit-lnl). 
*Note: These scripts are not part of the production vocal assistant pipeline. They are specifically designed for data collection, research, and thesis development.*

**Files and Tracked Metrics:**
* `Energy_Consumption.py`: Tracks energy consumption in Watts during model inference.
* `Hardware_Consumption.py`: Analyzes the load and utilization of the CPU/NPU.
* `RAM_Usage.py`: Monitors allocated memory peaks to identify hardware bottlenecks.
* `Power_Metrics.py`: Aggregates and logs power draw data.

---

<a id="results-discussion"></a>
## Results and Performance Discussion

This section presents the benchmarking results of the modular framework. To ensure statistical significance and mitigate transient hardware fluctuations, the evaluation is based on **over 1,000 experimental runs** for each model-hardware configuration.

### Methodology and Datasets
The evaluation focuses on accuracy, latency, and efficiency across three critical dimensions of the conversational pipeline:

* **Speech-to-Text (STT) & Text-to-Speech (TTS):** Benchmarked using the **LibriSpeech** dataset. Key metrics include *Real-Time Factor (RTF)* for latency and *Word Error Rate (WER)* for transcription integrity.
* **Small Language Modeling (SLM):** Tested via the **SQuAD (Stanford Question Answering Dataset)**. Performance was quantified through *Generation Throughput (Tokens/s)* and *Semantic Accuracy*.
* **System Telemetry:** Power consumption (Joules) and resource utilization (CPU/GPU/RAM) were recorded to identify efficiency patterns.

### Hardware Environment & Optimization
All tests were performed on an **Intel® Core™ Ultra SoC** architecture. To maximize edge performance and energy efficiency, the framework leverages the **OpenVINO™ toolkit** for hardware-specific optimizations. This allows for dynamic orchestration and offloading of workloads across the **CPU, iGPU, and NPU**, enabling the framework to support up to 3,388 unique execution pipelines.

The following tables summarize the mean results and variance for each category.

### 🎙️ Speech-to-Text (STT) Benchmark
*Dataset: LibriSpeech*

| Model (Params/M) | Device | RTF Mean | RTF Var | WER Mean | WER Var |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Moonshine Tiny (27) | CPU | 0.036 | 0.0000 | 0.069 | 0.0117 |
| Whisper Tiny (39) | GPU | 0.060 | 0.0002 | 0.090 | 0.0175 |
| Whisper Tiny (39) | CPU | 0.087 | 0.0007 | 0.090 | 0.0175 |
| Moonshine Base (61) | CPU | 0.064 | 0.0002 | 0.051 | 0.0089 |
| Distil Whisper Small (166) | GPU | 0.071 | 0.0002 | 0.053 | 0.0084 |
| Distil Whisper Small (166) | CPU | 0.290 | 0.0169 | 0.053 | 0.0084 |
| Whisper Base (74) | GPU | 0.075 | 0.0003 | 0.070 | 0.0109 |
| Whisper Base (74) | CPU | 0.143 | 0.0026 | 0.070 | 0.0109 |
| Distil Whisper Large V3 (756) | GPU | 0.098 | 0.0009 | 0.037 | 0.0057 |
| Distil Whisper Large V3 (756) | CPU | 1.240 | 0.4759 | 0.037 | 0.0057 |
| Whisper Small (244) | GPU | 0.100 | 0.0005 | 0.048 | 0.0077 |
| Whisper Small (244) | CPU | 0.405 | 0.0301 | 0.048 | 0.0077 |
| Whisper Large V3 Turbo (756) | GPU | 0.110 | 0.0010 | 0.042 | 0.0087 |
| Whisper Large V3 Turbo (756) | CPU | 1.386 | 0.5606 | 0.042 | 0.0087 |
| Qwen3-ASR (600) | GPU | 0.127 | 0.0022 | 0.037 | 0.0058 |
| Qwen3-ASR (600) | CPU | 0.200 | 0.0019 | 0.037 | 0.0058 |
| Fun-ASR-Nano (800) | GPU | 0.162 | 0.0014 | 0.026 | 0.0042 |
| Fun-ASR-Nano (800) | CPU | 0.144 | 0.0012 | 0.026 | 0.0042 |
| Whisper Medium (769) | GPU | 0.167 | 0.0015 | 0.044 | 0.0097 |
| Whisper Medium (769) | CPU | 1.281 | 0.3571 | 0.044 | 0.0097 |

<br>

### 🗣️ Text-to-Speech (TTS) Benchmark
*Dataset: LibriSpeech*

| Model (Params/M) | Device | RTF Mean | RTF Var |
| :--- | :---: | :---: | :---: |
| Piper TTS (20) | CPU | 0.034 | 0.0001 |
| Kitten TTS Nano (15) | CPU | 0.169 | 0.0000 |
| Pocket TTS (100) | CPU | 0.215 | 0.0001 |
| Kokoro TTS (82) | GPU | 1.330 | 0.3766 |
| Kokoro TTS (82) | CPU | 0.255 | 0.0054 |
| Supertonic 2 (66) | CPU | 0.285 | 0.0148 |
| Kitten TTS Mini (80) | CPU | 0.362 | 0.0004 |
| Soprano 1.1 (80) | CPU | 0.415 | 0.0073 |
| Qwen3 TTS (600) | GPU | 1.933 | 0.2755 |
| Qwen3 TTS (600) | CPU | 2.222 | 0.4743 |
| VoxCPM (500) | CPU | 5.670 | 0.2236 |
| OuteTTS 0.1 (350) | CPU | 37.852 | 4.673 |

<a id="getting-started"></a>
## 🚀 Getting Started

### 1. Environment Setup
Due to the vast amount of supported inference engines, it is highly recommended to use a dedicated virtual environment.

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Some specific engines like `llama_cpp` or `openvino` might require C++ build tools or specific hardware drivers depending on your OS).*

### 3. Download Local AI Models
To run the framework completely offline, you need to download the pre-compiled weights and configuration files for the models.

* Download all the required models from this [Google Drive Link](https://drive.google.com/drive/folders/1xDNm1kooOtHcSOL2VqS6_wEpoEkd5Paz?usp=drive_link).
* Extract the folders and place them in the correct root or `Models` directory as expected by the paths inside `DefMain.py`.

### 4. Running the Testbench
To launch the core evaluation framework:
```bash
python DefMain.py
```
Follow the interactive CLI to select your desired mode, choose the models from the library, target your hardware (CPU/GPU/NPU), and let the framework generate the performance dashboard!
