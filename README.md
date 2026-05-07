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

This directory contains the scripts used for benchmarking and analyzing the performance of AI models running on Edge hardware (e.g., Khadas Mind). 
*Note: These scripts are not part of the production vocal assistant pipeline. They are specifically designed for data collection, research, and thesis development.*

**Files and Tracked Metrics:**
* `Energy_Consumption.py`: Tracks energy consumption in Watts during model inference.
* `Hardware_Consumption.py`: Analyzes the load and utilization of the CPU/NPU.
* `RAM_Usage.py`: Monitors allocated memory peaks to identify hardware bottlenecks.
* `Power_Metrics.py`: Aggregates and logs power draw data.

---

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

* Download all the required models from this [Google Drive Link]([https://drive.google.com/drive/folders/1xDNm1kooOtHcSOL2VqS6_wEpoEkd5Paz?usp=drive_link]).
* Extract the folders and place them in the correct root or `Models` directory as expected by the paths inside `DefMain.py`.

### 4. Running the Testbench
To launch the core evaluation framework:
```bash
python DefMain.py
```
Follow the interactive CLI to select your desired mode, choose the models from the library, target your hardware (CPU/GPU/NPU), and let the framework generate the performance dashboard!
