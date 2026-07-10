FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch wheel: the default PyPI torch build bundles CUDA runtime libraries that are
# both unnecessary (this demo has no GPU access) and would blow well past the 10GB container
# budget on their own.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/
COPY ARCHITECTURE.md CLAUDE.md README.md ./

RUN chmod +x scripts/run_demo.sh

CMD ["bash", "scripts/run_demo.sh"]
