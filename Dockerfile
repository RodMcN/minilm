FROM nvidia/cuda:13.1.2-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

WORKDIR /workdir

COPY pyproject.toml uv.lock .python-version ./
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv sync --locked --no-install-project

COPY tokeniser.json .

COPY src/ src/

RUN uv sync --locked

# ENTRYPOINT ["uv", "run", "torchrun", \
ENTRYPOINT ["/workdir/.venv/bin/torchrun", \
    "--standalone", \
    "--nproc-per-node=gpu", \
    "--module", "src.minilm.training.train"]

#   docker run -it --rm \
#     --user "$(id -u):$(id -g)" \
#     --gpus all \
#     -v "$(pwd)/data:/workdir/data" \
#     -v "$(pwd)/runs:/workdir/runs" \
#     -v "$(pwd)/config.yaml:/workdir/config.yaml:ro" \
#     train 123 /workdir/config.yaml