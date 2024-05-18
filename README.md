<div align="center">

LongICLEval4Llama3
===========================
<h4> Long In-context Learning Evaluation on Llama3-70B-Instruct</h4>

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/CUDA-12.1.0-green)](https://developer.nvidia.com/cuda-downloads)
[![TensorRT-LLM](https://img.shields.io/badge/TRTLLM-v0.9.0-green)](https://github.com/NVIDIA/TensorRT-LLM)
[![tensorrtllm_backend](https://img.shields.io/badge/Backend-v0.9.0-green)](https://github.com/triton-inference-server/tensorrtllm_backend)
[![tritonserver](https://img.shields.io/badge/tritonserver-24.04-blue)](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-24-04.html#rel-24-04)
[![Medium](https://img.shields.io/badge/-@tinge.q-black?style=flat&logo=Medium&logoColor=white)](https://medium.com/@tinge.q)

---
<div align="left">

## Introduction
Llama3 by default has the 8K context window, I am curious about his capability for long in-context learning. 
There already exists several works about evaluation on this topic, including In-Context Learning with Long-Context Models: An In-Depth Exploration 
and Long-context LLMs Struggle with Long In-context Learning. 
I picked up the common test dataset Banking-77 both included in these two papers, and do the evaluation on Llama3–70B-Instruct model.

## Process flow
<img src="/resources/basic_process_flow.png" width="680"><BR>
 1. bash -x scripts/build_engine.sh
 1. bash -x scripts/config_repo.sh
 1. bash -x scripts/start_service.sh
 1. pip install -r requirements.txt
 1. python banking77_eval.py

