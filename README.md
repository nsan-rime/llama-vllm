# llama-vllm

Barebones Llama3B + vLLM setup for benchmarking.

## Set up

### MWE (local inference)

```bash
# Start Docker container and attach via VS Code
docker compose run -d dev-gpu

# Inside the container
huggingface-cli login --token=TOKEN_FOR_PULLING_ORPHEUS

# Run MWE inference locally
python truss/run-00-mwe-local.py
```

### MWE (Baseten inference)

```bash
# Start Docker container and attach via VS Code
docker compose run -d dev-gpu

# config.yaml is in /workspace/truss
# Assuming hf_orpheus_access_token_nay for pulling orpheus is configured properly on Baseten
cd /workspace/truss

# Enter truss API key for truss push
truss login
truss push

# Usage: python truss/run-mwe-01-baseten.py URL API_KEY_WITHOUT_PREFIX TEXT
python truss/run-01-mwe-baseten.py \
    https://model-zq8d5rdw.api.baseten.co/development/predict \
    API_KEY_WITHOUT_PREFIX \
    "this is a test"
```
