## 🧩 Modular Framework

This folder contains the core source code for the **Edge AI Conversational Framework**. Designed with modularity and edge-efficiency in mind, the architecture allows for seamless swapping of models and hardware targets.

### 🛠️ Key Components:
* **Dynamic Hardware Routing (`DefMain.py`):** The main orchestrator that handles the integration with the **OpenVINO™ toolkit**. It dynamically dispatches STT, SLM, and TTS inference workloads across the CPU, iGPU, and NPU, supporting up to **3,388 unique execution pipelines**.
* **Telemetry & Benchmarking:** Built-in profiling tools to track Real-Time Factor (RTF), Word Error Rate (WER), peak RAM utilization, and power consumption (Joules) across all inference pipelines.

---

### 🚀 Getting Started
Ready to test the framework on your own local hardware? Check out the full installation and execution guide here: 
👉 **[Insert Link to "How to Run It" Section Here]**
