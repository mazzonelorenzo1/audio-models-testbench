# 🧠 Edge AI Vocal Assistant (Core)

This directory contains the source code for the vocal assistant, highly optimized for local, offline execution.

The system is designed with a highly modular architecture, allowing developers to easily swap out the Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) engines.

## System Architecture
* **`ModularPipeline.py`**: Contains all the audio and slm models tested, it allows interchangability between models and five different testing modalities: 
* **`demo.py`**: The main entry point of the application. It spawns concurrent threads, manages the asynchronous text/audio buffer flow between modules, and handles user interaction via microphone and speakers.

## Getting Started
Ensure you have installed all the dependencies from the `requirements.txt` file and downloaded the necessary model weights. 

From the root directory of the project, run:
```bash
python src/demo.py
