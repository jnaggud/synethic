# Contributing

The supported entry points are `train.py` and `generate.py`. Scripts in `experiments/` are historical references with separate dependencies and known limitations.

## Local checks

Use Python 3.10 in a virtual environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
ruff check .
ruff format --check .
python -m pytest -q
```

Tests use small random fixtures and temporary directories. They do not require chest X-rays, network access, or a GPU. Add regression coverage when changing data handling, losses, checkpoint compatibility, or inference. Use `ruff format .` to apply formatting; archived experiments are excluded.

For changes to the command-line workflow, also run the [README demo](README.md#run-the-demo) with new run and output directories. GitHub Actions repeats those commands and retains a small `smoke-demo` artifact containing fixture samples, configuration, and metrics.

## Reports and changes

Open a [GitHub issue](https://github.com/jnaggud/synethic/issues) with the command, expected and actual behavior, error text, Python/TensorFlow versions, and a minimal reproduction. Keep source images, training checkpoints, credentials, and local paths out of public reports when they contain sensitive information.

Describe the behavior a pull request changes and the validation performed. Update the usage guide when options or output formats change. Distinguish execution checks from image-quality or privacy evidence; claims about model quality need recorded experiments.

## Documentation figures

The historical grids are preserved without enhancement. To rebuild their labeled presentation layouts:

```bash
python -m pip install -r requirements-figures.txt
python scripts/build_portfolio_assets.py
```

See [DATA.md](docs/DATA.md) for attribution and licensing of the source data and documentation figures.
