# 🧪 Edge AI Audio Models Testbench & Evaluation Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-blueviolet)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005ced)

## 📑 Table of Contents
- [Ownership & License](#licence)
- [Abstract](#abstract)
- [Operational Modes & Test Methodology](#operational-modes)
- [Supported Models Library](#supported-models)
- [Tools & Utility Scripts](#tools-and-utilities)
- [Hardware Profiling Benchmarks](#hardware-benchmarks)
- [Results and Performance Discussion](#results-discussion)
- [Getting Started](#getting-started)
- [Live Demo](#livedemo)

---

<a id="licence"></a>
## 📄 Ownership & License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

You are free to:
* **Share** - copy and redistribute the material in any medium or format.
* **Adapt** - remix, transform, and build upon the material.

Under the following terms:
* **Attribution (BY)** - You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* **NonCommercial (NC)** - You may not use the material for commercial purposes.
* **ShareAlike (SA)** - If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

To view a full copy of this license, visit: [http://creativecommons.org/licenses/by-nc-sa/4.0/](http://creativecommons.org/licenses/by-nc-sa/4.0/)

Owners:
* Lorenzo Mazzone, System research and applications
* Danilo Pau, System Research and Applications (danilo.pau@st.com)
* STMicrelectronics SRL

---

<a id="abstract"></a>
## 📖 Abstract
This repository provides a highly modular, end-to-end benchmarking framework designed to evaluate and compare locally hosted AI models for Conversational Voice Assistants. 

It allows developers to plug-and-play different **Speech-to-Text (STT)**, **Large Language Models (LLM)**, and **Text-to-Speech (TTS)** engines, rigorously profiling their performance on Edge hardware (e.g., Mini PCs, SBCs) without relying on cloud APIs, enabling the user to try and test up to 3,388 unique execution pipelines.


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
* **Whisper Family (OpenVINO):** [Tiny OV](https://huggingface.co/openai/whisper-tiny), [Base OV](https://huggingface.co/openai/whisper-base), [Small OV](), [Medium OV](https://huggingface.co/openai/whisper-small), [V3 Turbo OV](https://huggingface.co/openai/whisper-large-v3-turbo) `[CPU, GPU]`, [OpenVino wrapper](https://docs.openvino.ai/2024/notebooks/whisper-asr-genai-with-output.html)
* **Distil-Whisper (OpenVINO):** [Small](https://huggingface.co/distil-whisper/distil-small.en), [Large v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) `[CPU, GPU]` [OpenVino wrapper](https://docs.openvino.ai/2024/notebooks/distil-whisper-asr-with-output.html)
* **Moonshine:** [Tiny (27M)](https://huggingface.co/UsefulSensors/moonshine-tiny), [Base (61M)](https://huggingface.co/UsefulSensors/moonshine-base) `[CPU]`
* **Fun-ASR-Nano:** [Native](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) `[CPU]` and [OpenVINO Optimized](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/funasr-nano/funasr-nano.ipynb) `[CPU, GPU]`
* **Qwen3-ASR 0.6B:** [OpenVINO Optimized](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen3-asr/qwen3-asr.ipynb) `[CPU, GPU]`
* **Kyutai STT:** [Native](https://huggingface.co/docs/transformers/main/model_doc/stt) `[CPU]`

### 🧠 Large Language Models (LLMs)
* **Qwen 2.5 1.5B Instruct:** [OpenVINO int4](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov) `[CPU, GPU, NPU]`
* **DeepSeek R1 1.5B Distill:** [OpenVINO int4](https://huggingface.co/OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov) `[CPU, GPU, NPU]`
* **Qwen OV:** [Standard OpenVINO](https://huggingface.co/OpenVINO/Qwen3-0.6B-fp16-ov) `[CPU, GPU, NPU]`
* **SmolLM2 360M:** [Native](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) `[CPU]`
* **Gemma 3 270M:** [Native](https://huggingface.co/google/gemma-3-270m) `[CPU]`

### 🔊 Text-to-Speech (TTS) Engines
* **Kokoro:** [OpenVINO (`af_heart`)](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/kokoro/kokoro.ipynb) `[CPU, GPU, NPU]` and [Native](https://huggingface.co/hexgrad/Kokoro-82M) `[CPU]`
* **Piper:** [ONNX Runtime](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) `[CPU]`
* **KittenTTS:** [Nano 0.2](https://huggingface.co/KittenML/kitten-tts-nano-0.2) and [Mini 0.8 (ONNX)](https://huggingface.co/KittenML/kitten-tts-mini-0.8) `[CPU]`
* **Soprano 1.1 80M:** [Native](https://huggingface.co/ekwek/Soprano-1.1-80M) `[CPU]`
* **Pocket-TTS (Kyutai):** [Native](https://huggingface.co/kyutai/pocket-tts) `[CPU]`
* **Supertonic-2:** [ONNX Runtime](https://huggingface.co/Supertone/supertonic-2) `[CPU]`
* **OuteTTS 0.1:** [350M LLaMa-based](https://huggingface.co/OuteAI/OuteTTS-0.1-350M) `[CPU]`
* **Qwen3-TTS 0.6B:** [OpenVINO Optimized](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen3-tts/qwen3-tts.ipynb) `[CPU, GPU]`
* **VoxCPM 0.5B:** [Diffusion-based](https://huggingface.co/openbmb/VoxCPM-0.5B) `[CPU]`

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
All tests were performed on an **Intel® Core™ Ultra SoC** architecture. To maximize edge performance and energy efficiency, the framework leverages the **OpenVINO™ toolkit** for hardware-specific optimizations. This allows for dynamic orchestration and offloading of workloads across the **CPU, iGPU, and NPU**.
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

### 📊 Key Insights and Architectural Analysis

The benchmark results reveal crucial insights into how different architectures and hardware targets interact in an Edge computing environment:

#### 🎙️ Speech-to-Text (STT) Insights
* **The Padding Bottleneck:** While parameter count affects latency, architecture is more decisive. Modern models (like Moonshine) outperform the Whisper baseline by dropping fixed 30-second padding. By using variable-length attention, they dynamically scale computation to the exact audio length.
* **The Power of Distillation:** Distil-Whisper proves that aggressively pruning the autoregressive decoder while keeping the acoustic encoder maintains excellent accuracy (WER) while drastically cutting latency.
* **CPU vs. GPU Memory Limits:** Standard Transformer models bottleneck heavily on CPUs due to KV-cache memory bandwidth limits. GPU offloading fixes this. However, models utilizing Grouped Query Attention (GQA), like Qwen3-ASR, drastically reduce memory footprint and run highly effectively on the CPU alone.
* **The Non-Autoregressive (NAR) Anomaly:** FunASR-Nano actually ran *slower* on the GPU than on the CPU. Because its NAR architecture predicts all tokens in a single parallel pass, the time taken to move data to the GPU (dispatch overhead) exceeds the actual computation time, making the CPU the optimal target.

#### 🗣️ Text-to-Speech (TTS) Insights
* **The Efficiency Champion:** Piper TTS (RTF 0.034) proves that tiny models (20M) excel when paired with low-level execution. By exporting to ONNX and using native C++ vectorization, it completely bypasses Python overhead, achieving near-zero latency.
* **Smart Mid-Weight Architectures:** Pocket TTS (100M) outperformed smaller discrete models by operating in a continuous latent space at a low generation frequency. Similarly, Supertonic 2 makes diffusion models viable on the edge strictly through aggressive trajectory distillation (fewer inference steps).
* **The ALM CPU Bottleneck:** Audio Language Models (ALMs) like OuteTTS or VoxCPM are currently incompatible with CPU-only real-time constraints. Their autoregressive nature requires high-frequency sequential forward passes that immediately saturate CPU memory bandwidth, mandating dedicated hardware acceleration to be usable.

### ⚡ Energy Efficiency and Hardware Telemetry

To evaluate the viability of these models for Edge devices, it is essential to consider their energy footprint and memory utilization. Power consumption is inherently tied to the Real-Time Factor (RTF): models that execute inference rapidly minimize the time the System-on-Chip (SoC) spends in a high-power active state. These metrics where acquired via the HWiNFO application.

*Baseline System Idle state: 4.08 W Total System Power, 29.7% RAM.*

<br>

#### 🔋 STT Power Consumption & Energy Footprint
*Energy / s Audio is calculated as Average System Power × RTF. Lower is better.*

| Model | Device | CPU Pwr (W) | iGPU Pwr (W) | System Pwr (W) | Peak RAM (%) | Energy / s Audio (Joules) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Moonshine Tiny | CPU | 17.24 | - | 22.26 | 32.70 | **0.80** |
| Moonshine Base | CPU | 17.00 | - | 21.81 | 32.90 | **1.40** |
| Whisper Tiny | GPU | 18.11 | 1.92 | 23.43 | 40.10 | **1.41** |
| Distil-Whisper Small | GPU | 19.76 | 3.49 | 25.76 | 36.20 | **1.83** |
| Whisper Base | GPU | 19.65 | 2.48 | 25.67 | 41.60 | **1.93** |
| Whisper Tiny | CPU | 24.31 | - | 29.67 | 35.30 | **2.58** |
| Whisper Small | GPU | 21.56 | 4.53 | 27.69 | 45.70 | **2.77** |
| Distil-Whisper Large-V3 | GPU | 23.46 | 5.57 | 30.07 | 47.10 | **2.95** |
| Whisper Large-V3 Turbo | GPU | 22.53 | 6.39 | 29.42 | 51.80 | **3.24** |
| Qwen3-ASR | GPU | 26.53 | 6.54 | 33.40 | 47.00 | **4.24** |
| FunASR Nano | GPU | 21.29 | 3.84 | 27.01 | 48.00 | **4.38** |
| Whisper Base | CPU | 27.24 | - | 32.58 | 42.70 | **4.66** |
| FunASR Nano | CPU | 30.14 | - | 36.21 | 49.90 | **5.21** |
| Whisper Medium | GPU | 25.88 | 6.22 | 33.02 | 58.40 | **5.51** |
| Qwen3-ASR | CPU | 30.07 | 0.01 | 36.19 | 50.10 | **7.24** |
| Distil-Whisper Small | CPU | 30.33 | - | 35.68 | 37.10 | **10.35** |
| Whisper Small | CPU | 30.68 | - | 37.48 | 46.50 | **15.18** |
| Distil-Whisper Large-V3 | CPU | 30.11 | - | 37.51 | 49.90 | **46.51** |
| Whisper Medium | CPU | 30.97 | - | 38.68 | 57.20 | **49.55** |
| Whisper Large-V3 Turbo | CPU | 31.18 | - | 38.01 | 54.50 | **52.68** |

<br>

#### 🔋 TTS Power Consumption & Energy Footprint
*Energy / s Audio is calculated as Average System Power × RTF. Lower is better.*

| Model | Device | CPU Pwr (W) | iGPU Pwr (W) | System Pwr (W) | Peak RAM (%) | Energy / s Audio (Joules) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Piper TTS | CPU | 30.00 | 0.01 | 36.35 | 38.40 | **1.24** |
| Kitten TTS Nano | CPU | 20.67 | 0.00 | 25.08 | 36.30 | **4.24** |
| Supertonic 2 | CPU | 16.67 | 0.00 | 20.67 | 36.40 | **5.89** |
| Pocket TTS | CPU | 23.60 | 0.01 | 28.18 | 37.00 | **6.06** |
| Soprano 1.1 | CPU | 16.73 | 0.00 | 20.94 | 38.50 | **8.69** |
| Kokoro TTS | CPU | 30.95 | 0.01 | 36.51 | 49.00 | **9.31** |
| Kitten TTS Mini | CPU | 21.11 | 0.00 | 25.81 | 38.20 | **9.34** |
| Kokoro TTS | GPU | 26.07 | 5.64 | 32.78 | 49.00 | **43.60** |
| Qwen3 TTS | GPU | 24.96 | 3.42 | 30.60 | 50.00 | **59.15** |
| Qwen3 TTS | CPU | 26.65 | 0.00 | 29.03 | 50.20 | **64.50** |
| VoxCPM | CPU | 15.90 | 0.00 | 20.07 | 44.80 | **113.80** |
| OuteTTS 0.1 | CPU | 26.85 | 0.00 | 32.48 | 47.20 | **1229.43** |

### 💡 Energy & System Bottlenecks: Final Takeaways

* **Speed Equals Efficiency:** Ultra-lightweight models (e.g., Moonshine, Piper) minimize energy use (~1 Joule/s) primarily because their low RTF allows the SoC to instantly return to its ~4W idle state.
* **The iGPU Advantage:** Hardware offloading via OpenVINO is mandatory for heavier models. Running Whisper Large on the iGPU drops energy consumption exponentially (e.g., from 52J to 3J) by drastically cutting inference time.
* **The Edge Trade-off (CPU vs. RAM):** While C++ optimized models run efficiently on the CPU, they often lock it at 100% utilization, risking OS starvation. Offloading to the iGPU solves this but shifts the bottleneck to the Unified Memory Architecture (UMA), pushing RAM usage up to 70% for a full conversational pipeline.

---

<a id="getting-started"></a>
## 🚀 Getting Started

### Prerequisites
Ensure you have **Python 3.10 or 3.11** installed on your system, along with `git`.

### 1. Clone the Repository
First, clone this repository to your local machine and navigate into the project directory:

```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
```

### 2. Environment Setup
Due to the vast amount of supported inference engines, it is highly recommended to use a dedicated virtual environment.

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Some specific engines like `llama_cpp` or `openvino` might require C++ build tools or specific hardware drivers depending on your OS).*

### 4. Download Local AI Models
To run the framework completely offline, you need to download the pre-compiled weights and configuration files for the models.

* Download all the required models from this [Google Drive Link](https://drive.google.com/drive/folders/1xDNm1kooOtHcSOL2VqS6_wEpoEkd5Paz?usp=drive_link).
* Extract the folders and place them in the correct root or `Models` directory as expected by the paths inside `DefMain.py`.
* A test audio is already present inside the [Google Drive Link](https://drive.google.com/drive/folders/1xDNm1kooOtHcSOL2VqS6_wEpoEkd5Paz?usp=drive_link). To start testing all the models, simply put it in the `Models` folder. If you want to change it, also remember to change the reference text in order to get accurate WER results.

Your folder structure should look like this:

```Plaintext
YourRepoName/
├── DefMain.py
├── requirements.txt
├── Models/
│   ├── [Model_Folder]
│   └── test_audio.wav  <-- Place the test audio here
├── Outputs
└── ...
```

### 4. Running the Testbench
To launch the core evaluation framework:
```bash
python DefMain.py
```
Follow the interactive CLI to select your desired mode, choose the models from the library, target your hardware (CPU/GPU/NPU), and let the framework generate the performance dashboard!

---

<a id="livedemo"></a>

## 🎬 Live Demo: The Optimal Edge Pipeline

Based on the extensive benchmarking and architectural analysis discussed above, we selected the top-performing STT ([Moonshine Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny)), SLM ([Qwen 2.5 Instruct](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov)), and TTS ([Piper](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)) models to build a fully functional, real-time conversational agent. 

This interactive demo showcases the true potential of the framework in a real-world scenario. It features the **Semantic Bridge Agent** for ultra-low latency, human-like prosody, and leverages **OpenVINO™** for optimal hardware acceleration across the Edge SoC.

You can explore the complete source code and run the interactive demo yourself at the following dedicated repository:

👉 **[Access the Live Demo Repository Here](https://github.com/mazzonelorenzo1/Conversational-Edge-Model)**
