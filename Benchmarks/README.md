# 📊 Audio Models Testbench

This directory contains the scripts used for benchmarking and analyzing the performance of AI models (LLM, TTS, STT) running on Edge hardware (e.g., Khadas Mind).

These scripts are **not** part of the production vocal assistant pipeline. They are specifically designed for data collection, research, and thesis development.

## Files and Tracked Metrics
* **`Energy_Consumption.py`**: Tracks energy consumption in Watts during model inference.
* **`Hardware_Consumption.py`**: Analyzes the load and utilization of the CPU/NPU.
* **`RAM_Usage.py`**: Monitors allocated memory peaks to identify hardware bottlenecks.
* **`Power_Metrics.py`**: Aggregates and logs power draw data.
