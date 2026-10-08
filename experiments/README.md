# Archived prototypes

These scripts preserve the project's original source work. Use `../train.py` for the supported, tested implementation. The archived scripts have not been certified to run with the root dependency set.

| File | Purpose | Known limitations |
| --- | --- | --- |
| `onepager.py` | Original 64×64 DCGAN experiment | Fixed dimensions; long default run; incomplete final batches are skipped; SSIM uses signed pixels. |
| `onepager_v2.py` | Configurable models and checkpoints | Creates the dataset directory before checking whether to download; recursively mixes input folders; tensor loops complicate graph-mode similarity loss. |
| `onepager_enhanced.py` | Logging and architecture exploration | Ends after model definitions; no complete training entry point. |
| `onepager_enhanced_gpu.py` | Metal and mixed-precision exploration | NumPy/skimage similarity operations conflict with graph tracing and break differentiability; the penalty is added twice. |

`requirements-legacy.txt` preserves the original dependency list. It is not the supported installation path. Historical comments describe design intent and may overstate properties that were never established; the root README and results page define the current project claims.

The historical epoch images in `docs/assets/` match the output naming and layout of `onepager.py`. The complete run log and environment were not retained, so exact reproduction of those historical images is not claimed. Old checkpoints are not compatible with the supported trainer.
