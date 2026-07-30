set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Remove generated Python and packaging artifacts.
clean:
    find . \
        -path './.venv' -prune \
        -o -type d \( \
            -name __pycache__ \
            -o -name '*.egg-info' \
            -o -name .pytest_cache \
            -o -name .ruff_cache \
            -o -name .mypy_cache \
            -o -name .ipynb_checkpoints \
        \) \
        -prune \
        -exec rm -rf -- {} +
    find . \
        -path './.venv' -prune \
        -o -type f \( \
            -name '*.py[co]' \
            -o -name '*$py.class' \
            -o -name .coverage \
        \) \
        -exec rm -f -- {} +
    rm -rf -- build dist wheels htmlcov

tb:
    uv run tensorboard --logdir=runs