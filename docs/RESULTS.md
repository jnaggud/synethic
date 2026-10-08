# Results and verification

[← Project overview](../README.md)

## Historical image generation

![Saved grids from epochs 1, 250, and 500](assets/training-progression.png)

The three grids are existing project artifacts copied without alteration from `poc_images/image_at_epoch_0001.png`, `image_at_epoch_0250.png`, and `image_at_epoch_0500.png`. They show a transition from repetitive texture to recognizable chest-like structure, alongside artifacts and limited apparent diversity.

The epoch labels come from the original filenames. The complete run log, exact dependency snapshot, and quantitative evaluation for that run were not retained. The saved grids match the naming and layout used by the archived `onepager.py` script. They are historical evidence of prototype output, not a benchmark of the supported trainer or proof that it reproduces the same images.

The public trainer is derived from the configurable v2 approach, with corrected data loading, TensorFlow-native SSIM, explicit configuration, smaller default networks, and tested checkpoint handling. Old checkpoints are not distributed and cannot be loaded into this trainer.

## Public-release verification

Verified locally on 8 October 2026 with Python 3.10.13, TensorFlow 2.16.2, Keras 3.12.4, and CPU execution on macOS / Apple Silicon:

| Check | Result |
| --- | --- |
| Automated tests | 15 passed |
| Lint and formatting | Passed for the supported trainer, inference CLI, tests, and figure script |
| Dependency consistency | `pip check` passed |
| JPEG and PNG preprocessing | Correct grayscale shape and normalization verified |
| Train/test separation | Nested held-out folders are not loaded by the training loader |
| Similarity gradient | Finite, nonzero gradients verified in a `tf.function` |
| Training update | Generator weights changed after one training step |
| Checkpoint restore | Model outputs, optimizer iterations, epoch, reference images, and noise-generator state restored |
| CLI smoke test | Fresh run and resume verified; overwrite and configuration mismatch rejected |
| Checkpoint inference | Restored generator reproduces the saved fixed-noise grid without loading the dataset |
| Standalone model export | Reloaded Keras model matches checkpoint-generator outputs on independent noise inputs |
| Inference input validation | Missing metadata/checkpoints, unmatched generator weights, invalid counts, and output overwrite are rejected |
| Actual X-ray input | One training step completed with the documented eight-image data check |

The actual-image check used 64×64 inputs, batch size 2, generator width 64, discriminator width 8, and one step. Reported losses were generator **1.3784**, discriminator **1.0582**, and similarity penalty **0.000000**. A zero penalty means no selected comparison exceeded the threshold during that step; it is not a privacy result.

These are execution and regression checks. They do not measure model quality or demonstrate convergence. CI repeats the dataset-free CPU checks on Linux, runs the README training/export workflow, and retains fixture outputs and logs in its `smoke-demo` artifact for 14 days. The current status is visible in [GitHub Actions](https://github.com/jnaggud/synethic/actions/workflows/checks.yml).

## What remains unvalidated

- Clinical realism, diagnostic utility, and downstream task benefit.
- Dataset-wide memorization, membership inference, and formal privacy properties.
- FID or other generative-model quality metrics.
- Performance comparisons across CPU, Metal, and CUDA hardware.
- Exact reproducibility of the historical 500-epoch output.

The SSIM implementation follows the input-range requirements in [TensorFlow's SSIM documentation](https://www.tensorflow.org/api_docs/python/tf/image/ssim). It checks only a fixed reference subset. A production research program would need held-out evaluation, broader similarity analysis, privacy attacks, and domain-expert review before drawing stronger conclusions.
