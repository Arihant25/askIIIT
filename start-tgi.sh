model=Qwen/Qwen3-1.7B
volume=$PWD/models # share a volume with the Docker container to avoid downloading weights every run

# Load environment variables
source .env

docker run --gpus all --shm-size 1g -p 8000:80 -v $volume:/models \
    -e HF_TOKEN=$HF_TOKEN \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id $model \
    --max-total-tokens 8093 \
    --max-batch-prefill-tokens 8092
    # --max-input-tokens 8092