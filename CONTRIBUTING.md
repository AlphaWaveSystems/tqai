# Contributing to tqai

Thank you for your interest in contributing to tqai! This guide explains how to get involved.

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/AlphaWaveSystems/tqai/issues) with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- tqai version (`tqai info`), OS, Python version
- Model ID and backend (torch/mlx) you were using

## Pull Request Flow

1. **Fork** the repository on GitHub
2. **Branch** from `main` — use a descriptive name (e.g., `fix/dtype-mismatch`, `feat/bit-packing`)
3. **Develop** your changes following the code style guidelines below
4. **Test** your changes (see Testing section)
5. **Submit a PR** against `main` with a clear description of what and why

Keep PRs focused on a single concern. If you have multiple unrelated changes, submit separate PRs.

## Code Style

- **Linting**: Run `ruff check .` before committing
- **Formatting**: Follow PEP 8, 100-char line length
- **Imports**: Sorted by ruff (isort compatible)
- **Types**: Use type hints for public API functions

## DCO Sign-Off

All commits must include a Developer Certificate of Origin sign-off line:

```
Signed-off-by: Your Name <your@email.com>
```

Add this automatically with:

```bash
git commit -s -m "Your commit message"
```

This certifies that you wrote or have the right to submit the code under the project's license. See [developercertificate.org](https://developercertificate.org/) for the full text.

## Testing

Before submitting a PR, run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit + accuracy tests
pytest tests/ --ignore=tests/test_e2e_models.py --ignore=tests/test_e2e_large_models.py

# Run linter
ruff check .
```

For changes to the quantizer or cache, also run the accuracy tests:

```bash
pytest tests/test_accuracy.py -v
```

## Adding Support for New Head Dimensions

If you need codebooks for a head dimension not in `{64, 96, 128, 256}`:

```bash
pip install tqai[codegen]
python scripts/generate_codebooks.py
```

## Architecture Overview

- `src/tqai/quantizer.py` — Core PolarQuantizer algorithm
- `src/tqai/backend/` — PyTorch + MLX abstraction (Protocol-based)
- `src/tqai/codebook/` — Lloyd-Max codebook generation and loading
- `src/tqai/cache/` — HuggingFace and mlx-lm cache wrappers
- `src/tqai/cli.py` — CLI tool
- `src/tqai/convert.py` — Offline model conversion

## License

By contributing to tqai, you agree that your contributions will be licensed under the MIT License.
