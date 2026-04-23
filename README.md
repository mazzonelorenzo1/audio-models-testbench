# 🧪 Edge AI Audio Models Testbench & Evaluation Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-blueviolet)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005ced)

<a id="abstract"></a>
## 📖 Abstract
This repository provides a highly modular, end-to-end benchmarking framework designed to evaluate and compare locally hosted AI models for Conversational Voice Assistants. 

It allows developers to plug-and-play different **Speech-to-Text (STT)**, **Large Language Models (LLM)**, and **Text-to-Speech (TTS)** engines, rigorously profiling their performance on Edge hardware (e.g., Mini PCs, SBCs) without relying on cloud APIs.

---

<a id="features"></a>
## ✨ Key Features & Profiling Metrics
The framework doesn't just run the models; it profiles the hardware and software efficiency in real-time. During execution, it automatically tracks and generates charts for:
- **Real-Time Factor (RTF):** To ensure audio generation/transcription is faster than the audio duration itself (RTF < 1.0).
- **Time-To-First-Token (TTFT):** LLM responsiveness measurement.
- **Word Error Rate (WER):** Accuracy tracking for STT models using the `jiwer` library.
- **Hardware Impact:** Real-time monitoring of Peak RAM usage (%) and CPU utilization (%) during inference.

---

<a id="supported-models"></a>
## 🧩 Supported Models
The architecture is strictly modular. The core `DefMain.py` script currently integrates and supports the following local models:

### 🎤 Speech-to-Text (STT)
- Whisper (via `pywhispercpp`)
- HuggingFace Seq2Seq Speech Models (via `optimum.intel` / OpenVINO)

### 🧠 Large Language Models (LLM)
- LLaMA variants (via `llama_cpp`)
- Qwen / Llama-3 (via `openvino_genai`)
- AutoModelForCausalLM (Standard HuggingFace)

### 🔊 Text-to-Speech (TTS)
- Piper TTS
- Kokoro (Both ONNX and PyTorch implementations)
- Moshi / Mimi
- KittenTTS & Supertonic
- Soprano TTS
- Pocket TTS

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

### 3. Running the Testbench
To launch the evaluation framework and generate the performance reports:
```bash
python DefMain.py
```
Upon completion, the script will output a comprehensive `matplotlib` dashboard displaying system resource impact, RTF curves, and WER scores across the different runs.

---

<a id="repository-structure"></a>
## 📁 Repository Structure
```text
edge-audio-testbench/
├── DefMain.py                # The core benchmarking orchestrator
├── requirements.txt          # Full list of dependencies
├── assets/                   # Evaluation charts and diagrams
└── [Other Utility Scripts]   # Helper scripts for dataset acquisition
```
