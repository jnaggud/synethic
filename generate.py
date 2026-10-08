"""Generate a sample grid or export a generator from a saved SynEthic run.

Training images are not needed. Smoke-test checkpoints produce fixture outputs,
not meaningful chest X-rays. Run ``python generate.py --help`` for options.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import tensorflow as tf

from train import Config, build_generator, save_grid


def load_generator(run_dir):
    """Restore generator weights and epoch without constructing training state."""
    run_dir = Path(run_dir)
    metadata_path = run_dir / "config.json"
    if not metadata_path.is_file():
        raise ValueError("Run directory must contain config.json from train.py")
    metadata = json.loads(metadata_path.read_text())
    config = Config(**metadata["config"])
    checkpoint_path = tf.train.latest_checkpoint(str(run_dir / "checkpoints"))
    if not checkpoint_path:
        raise ValueError("No checkpoint found in this run directory")
    generator = build_generator(config)
    epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(generator=generator, epoch=epoch)
    status = checkpoint.restore(checkpoint_path)
    # Training checkpoints also contain the discriminator, optimizers and references.
    # Those objects are deliberately omitted; every generator variable must match.
    try:
        status.expect_partial().assert_existing_objects_matched()
    except AssertionError as error:
        raise ValueError("Checkpoint does not match the saved generator configuration") from error
    return generator, config, metadata, int(epoch.numpy()), Path(checkpoint_path).name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing training run")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output folder")
    parser.add_argument(
        "--count", type=int, default=16, help="Images in the grid, 1–16 (default: 16)"
    )
    parser.add_argument("--seed", type=int, help="Noise seed (default: saved training seed)")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU use")
    parser.add_argument(
        "--export-model", action="store_true", help="Also save a standalone generator.keras model"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 16:
        parser.error("--count must be between 1 and 16")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty. Choose a new --output-dir")
    generator, config, run_metadata, epoch, checkpoint_name = load_generator(args.run_dir)
    seed = config.seed if args.seed is None else args.seed
    noise = tf.random.stateless_normal([args.count, config.noise_dim], seed=[seed, 0])
    images = generator(noise, training=False)
    tf.debugging.assert_all_finite(images, "Non-finite generator output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_grid(images, args.output_dir / "samples.png")
    if args.export_model:
        generator.save(args.output_dir / "generator.keras")
    metadata = {
        "checkpoint": checkpoint_name,
        "epoch": epoch,
        "seed": seed,
        "count": args.count,
        "smoke_test": run_metadata["smoke_test"],
        "config": asdict(config),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "exported_model": args.export_model,
    }
    (args.output_dir / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if run_metadata["smoke_test"]:
        print("Smoke-test checkpoint: fixture outputs, not a medical-imaging result.")
    print(f"Generated {args.count} samples from epoch {epoch} in {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, tf.errors.OpError) as error:
        raise SystemExit(str(error)) from error
