# Edge AI Vocal Assistant & Performance Testbench

This repository hosts a fully offline, end-to-end vocal assistant optimized for Edge devices (specifically the Khadas Mind 2 AI Maker Kit). The project focuses on modularity and performance optimization, featuring a custom chunking algorithm to minimize perceived latency in real-time conversations.

## 🚀 Key Features
* **Fully Offline**: STT, LLM, and TTS engines run locally without external API dependencies.
* **Modular Architecture**: Easily swap models (e.g., Moonshine vs. Whisper, Qwen vs. Llama).
* **Fully Functional**: A custom conversational demo with the three best performing models (Moonshine Tiny, Qwen 2.5 Instruct, Piper TTS).
* **Comprehensive Benchmarking**: Integrated tools to measure latency (RTF), accuracy (WER), hardware utilization (RAM/CPU/NPU), and power consumption.

---

## 📂 Repository Structure

You can explore the different components of the project through the links below:

* [**main/**](./main) - **Core Logic**: Contains the main pipeline, the interactive demo, and the Semantic Bridge Agent.
* [**benchmarks/**](./Benchmarks) - **Evaluation Suite**: Scripts for measuring performance, power draw, and hardware stress tests.
* [**tools/**](./Tools) - **Utilities**: Auxiliary scripts for model management and Hugging Face interactions.

---

## 🛠️ Tech Stack for the Demo
* **STT**: Moonshine Tiny (27M Parameters) 
* **LLM**: Qwen 2.5 Instruct (Optimized for Intel NPU via ONNX, 1.5B Parameters)
* **TTS**: Piper TTS (20M Parameters)

---

## 🏁 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed. It is highly recommended to use a virtual environment.

### 2. Installation
Clone the repository and install the required packages:
```bash
git clone [https://github.com/mazzonelorenzo1/audio-models-testbench-and-vocal-assistant.git](https://github.com/mazzonelorenzo1/audio-models-testbench-and-vocal-assistant.git)
cd your-repo-name
pip install -r requirements-demo.txt
pip install -r requirements-modular_pipeline.txt
```

### 3. Running the Assistant
To start the interactive voice conversation:
```bash
python src/demo.py
```

### 4. Running Benchmarks
To evaluate various model performances:
```bash
python src/Modular_Pipeline.py
```

## 📊 Evaluation Metrics
The project evaluates the pipeline using the following scientific metrics:
* **WER (Word Error Rate)**: For Speech-to-Text accuracy.
* **TTFT (Time To First Token)**: LLM response speed.
* **TTFA (Time To First Audio)**: Overall latency from user input to speech output.
* **RTF (Real-Time Factor)**: TTS efficiency.
* **Tk/s (Tokens per Second)**: LLM generation throughput.
* **Semantic Similarity**: To evaluate the accuracy and consistency of the LLM responses.
* **Power Efficiency**: Energy consumption (Watts) and hardware load (CPU/NPU/RAM) per inference cycle.

---

## 📝 License
This project is developed for academic and research purposes as part of a Master's Thesis for Milano-Bicocca University and ST Microelectronics.
