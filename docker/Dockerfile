FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch wheel: the default PyPI torch build bundles CUDA runtime libraries that are
# unnecessary in this container (embedding/retrieval run on CPU; LLM inference happens against
# the external ROCm/vLLM service) and would blow well past the 10GB image budget on their own.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/
COPY ARCHITECTURE.md README.md ./

RUN chmod +x scripts/run.sh

CMD ["bash", "scripts/run.sh"]
