# 🧠 Edge AI Vocal Assistant (Core)

This directory contains the source code for the vocal assistant, highly optimized for local, offline execution on the Khadas Mind 2 AI Maker Kit.

The system is designed with a highly modular architecture, allowing developers to easily swap out the Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) engines.

## System Architecture
* **`ModularPipeline.py`**: Houses the implementations for all evaluated audio and Small Language Models (SLMs). It provides seamless model interchangeability and supports five distinct evaluation modes: Full Pipeline (STT -> LLM -> TTS), Liquid ONNX Engine Benchmark, STT WER Evaluation, TTS RTF Evaluation, and LLM Tk/s & Semantic Similarity Evaluation.
* **`demo.py`**: The primary interactive application utilizing the three best-performing models: Moonshine Tiny, Qwen 2.5 Instruct, and Piper TTS. It enables seamless, real-time conversational capabilities running fully offline at the edge.

## Getting Started
Ensure you have installed all the dependencies from the `requirements.txt` file and downloaded the necessary model weights. 

From the root directory of the project, run:
```bash
python src/demo.py
