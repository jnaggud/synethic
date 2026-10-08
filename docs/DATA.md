# Data and attribution

The original prototype uses the chest X-ray portion of:

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification,” Mendeley Data, V2, doi: [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2).

The [publisher's dataset page](https://data.mendeley.com/datasets/rscbjbr9sj/2) lists **CC BY 4.0**. See the [license terms](https://creativecommons.org/licenses/by/4.0/). Attribution does not imply endorsement by the dataset authors.

## Obtaining the data

Download the chest X-ray archive from the dataset page, extract it locally, and locate `chest_xray/train/PNEUMONIA`. Archive layouts may include an additional enclosing directory. Pass the folder containing the JPEG files to `--data-dir`.

The supported loader does not automatically download archives and does not recursively combine splits. Choose a training directory deliberately, and keep held-out data separate for any subsequent evaluation.

## Changes and sample images

The training pipeline converts images to grayscale, resizes them (64×64 by default), and normalizes pixel values. The generator learns from that data and produces new model outputs from noise.

`docs/assets/epoch-001.png`, `epoch-250.png`, and `epoch-500.png` are unmodified copies of saved grids in the original project. Their filenames identify the recorded epoch. They are generated outputs, not source dataset images. Their inclusion does not establish that memorization is absent.

Documentation sample images and presentation layouts in `docs/assets/` are distributed under **CC BY 4.0**, with credit to Jefferson Duggan / SynEthic and the dataset authors above. The source-code MIT license does not replace the dataset's terms.

No source dataset, model weights, patient-level evaluation, or private business documentation is included in this repository.
