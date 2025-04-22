# First stage: base image with CUDA and Python
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Add same python dependencies as Baseten
RUN pip install \
    torch==2.5.1 \
    tokenizers==0.21 \
    asyncio==3.4.3 \
    transformers==4.50.0 \
    vllm==0.7.3 \
    fastapi==0.115.6 \
    uuid7==0.1.0

# Add truss for running truss push
RUN pip install --upgrade truss gradio

WORKDIR /workspace
