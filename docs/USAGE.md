# Usage guide

[← Project overview](../README.md)

## Setup

Use Python 3.10 and the pinned dependencies in `requirements.txt`. The README's [dataset-free demo](../README.md#run-the-demo) exercises the full training and inference path on CPU.

For real training, follow [DATA.md](DATA.md) and select a directory containing JPEG or PNG training images. Files are sorted by name; subdirectories, hidden files, and other extensions are ignored. RGB inputs become grayscale. Images are resized with bilinear interpolation and mapped to `[-1, 1]`.

## Check actual image inputs

Use a new run directory for each experiment:

```bash
python train.py \
  --data-dir chest_xray_data/chest_xray/train/PNEUMONIA \
  --run-dir runs/data-check \
  --cpu --max-images 8 --batch-size 2 --steps-per-epoch 1 \
  --gen-filters 64 --disc-filters 8
```

This selects the first eight filenames and runs one batch. It checks decoding and training on actual X-rays; one step is insufficient to assess image quality.

## Configure training

`python train.py --help` lists every option. The main defaults are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `1` | Additional epochs in this invocation, including on resume |
| `--image-size` | `64` | Output width and height; at least 16 and divisible by 16 |
| `--batch-size` | `16` | Training images per batch; the last partial batch is retained |
| `--noise-dim` | `100` | Generator input dimension |
| `--gen-filters` | `256` | Initial generator width; at least 16 and divisible by 16 |
| `--disc-filters` | `32` | Initial discriminator width |
| `--learning-rate` | `0.0002` | Learning rate for both Adam optimizers |
| `--privacy-weight` | `0.1` | Similarity term's contribution to generator loss |
| `--similarity-threshold` | `0.9` | SSIM threshold above which comparisons contribute a penalty |
| `--reference-size` | `16` | Maximum number of fixed training references |
| `--seed` | `42` | Initialization, shuffle, and sampling seed |

`--max-images` and `--steps-per-epoch` are optional limits for execution checks. `--cpu` disables GPU use. `--smoke-test` replaces input data with four deterministic random fixtures and overrides image size, batch size, model widths, and reference count with small values.

The first `reference-size` images in filename order form the reference subset. This selection is reproducible but may be unrepresentative, especially when filenames group patients. The [design notes](DESIGN.md#similarity-regularization) explain the implications.

## Resume a run

Repeat the original command with `--resume`, preserving the model and data options. For the README demo:

```bash
python train.py --smoke-test --cpu --run-dir runs/smoke --resume --epochs 2
```

This resumes the latest checkpoint and trains two additional epochs. It appends metrics and creates new sample grids. The last three checkpoints are retained. Existing nonempty directories require `--resume`; conflicting configurations are rejected.

Keep the original data directory and its contents unchanged. The configuration check compares the path and options, not file contents. Checkpoints are saved at epoch boundaries, so an interrupted epoch is repeated from the latest complete checkpoint. Seeds and saved noise state improve repeatability, but data-iterator and dropout state are not fully restored; resumed training is not promised to match an uninterrupted run bit for bit.

## Generate from a saved checkpoint

```bash
python generate.py \
  --run-dir runs/pneumonia \
  --output-dir runs/pneumonia-preview \
  --seed 123 --count 16 --cpu --export-model
```

The original dataset is not required. The command loads the latest checkpoint and saved architecture, then runs the generator in inference mode. It writes:

- `samples.png`: up to 16 grayscale images in a four-column grid. Unused cells in the last row are black.
- `generation.json`: checkpoint name, epoch, seed, sample count, model options, and runtime versions.
- `generator.keras`: a standalone model, when `--export-model` is supplied.

Omitting `--seed` uses the training seed and, for 16 images, recreates the checkpoint's fixed-noise grid on the same runtime. Use a new output directory for each generation request; existing files are protected from overwrite.

### Use the exported model

The export contains only the generator. It excludes the discriminator, optimizers, and explicit reference images stored in training checkpoints.

```python
import tensorflow as tf

model = tf.keras.models.load_model("runs/preview/generator.keras", compile=False)
noise_dim = model.input_shape[-1]
noise = tf.random.stateless_normal([4, noise_dim], seed=[123, 0])
images = model(noise, training=False)  # [4, height, width, 1], values in [-1, 1]
```

Training checkpoints contain real reference images and should be treated as training data. A generator-only export removes those explicit references, but does not establish that the model has not memorized training data.

## Inspect a run

`metrics.csv` records batch-size-weighted mean generator loss, discriminator loss, similarity penalty, epoch duration, and step count. A zero similarity penalty means no selected comparison crossed the threshold. GAN losses do not measure image quality, and shorter epochs limited by `--steps-per-epoch` are not comparable to full passes over the dataset.

The same fixed-noise vectors are used for each epoch's grid, making visual changes easier to inspect. Inference metadata marks smoke-test outputs explicitly. Configuration and environment logs live alongside the outputs for later comparison.

## Apple Silicon

CPU is the verified local path. Optional Metal acceleration uses:

```bash
python -m pip install -r requirements-metal.txt
```

Omit `--cpu` to let TensorFlow use an available device. Metal performance and exact cross-device reproducibility have not been benchmarked. The supported trainer uses float32.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| No JPEG/PNG images | Point at the class directory containing files, not the parent `train/` or dataset root. |
| Run/output directory is not empty | Choose a new directory, or use `--resume` for an existing training run. |
| Resume options must match | Reuse the options in that run's `config.json`, including input path and limits. |
| No checkpoint found | A training epoch must finish before inference or resume is available. |
| GPU initialization or memory errors | Retry the dataset-free demo with `--cpu`; reduce batch size or model widths in a new run. |

For bug reports, include the command, error, dependency versions, and a dataset-free reproduction when possible. [Contribution guide →](../CONTRIBUTING.md)
