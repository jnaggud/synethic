"""SynEthic: a DCGAN with an experimental reference-similarity penalty.

Run ``python train.py --help`` for training and dataset-free smoke-test options.
The similarity penalty is a heuristic, not a privacy guarantee.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers


@dataclass(frozen=True)
class Config:
    image_size: int = 64
    batch_size: int = 16
    noise_dim: int = 100
    gen_filters: int = 256
    disc_filters: int = 32
    learning_rate: float = 0.0002
    privacy_weight: float = 0.1
    similarity_threshold: float = 0.9
    reference_size: int = 16
    seed: int = 42

    def __post_init__(self):
        if self.image_size < 16 or self.image_size % 16:
            raise ValueError("image-size must be at least 16 and divisible by 16")
        for name in ("batch_size", "noise_dim", "disc_filters", "reference_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name.replace('_', '-')} must be positive")
        if self.gen_filters < 16 or self.gen_filters % 16:
            raise ValueError("gen-filters must be at least 16 and divisible by 16")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning-rate must be finite and positive")
        if not math.isfinite(self.privacy_weight) or self.privacy_weight < 0:
            raise ValueError("privacy-weight must be finite and nonnegative")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity-threshold must be between 0 and 1")


def load_dataset(data_dir, config, max_images=None):
    """Read one image directory without mixing train/test subdirectories."""
    if max_images is not None and max_images < 1:
        raise ValueError("max-images must be positive")
    directory = Path(data_dir)
    paths = sorted(
        str(p)
        for p in directory.glob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError(
            f"No JPEG/PNG images in {directory}. Point --data-dir directly at "
            "a training class folder, e.g. chest_xray_data/chest_xray/train/PNEUMONIA."
        )
    paths = paths[:max_images]

    def preprocess(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
        image.set_shape([None, None, 1])
        image = tf.image.resize(image, [config.image_size, config.image_size])
        return tf.cast(image, tf.float32) / 127.5 - 1.0

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset, len(paths)


def build_generator(config):
    size = config.image_size // 16
    model = tf.keras.Sequential(name="generator")
    model.add(layers.Input(shape=(config.noise_dim,)))
    model.add(layers.Dense(size * size * config.gen_filters, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Reshape((size, size, config.gen_filters)))
    for divisor in (2, 4, 8, 16):
        model.add(
            layers.Conv2DTranspose(
                config.gen_filters // divisor, 5, strides=2, padding="same", use_bias=False
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Conv2DTranspose(1, 5, padding="same", activation="tanh"))
    return model


def build_discriminator(config):
    model = tf.keras.Sequential(name="discriminator")
    model.add(layers.Input(shape=(config.image_size, config.image_size, 1)))
    for multiplier in (1, 2, 4, 8):
        model.add(layers.Conv2D(config.disc_filters * multiplier, 5, strides=2, padding="same"))
        if multiplier > 1:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(negative_slope=0.2))
        model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


class PrivacyGuardian(tf.Module):
    """Penalize high SSIM against a small, fixed reference subset.

    Every operation stays in TensorFlow to preserve graph-mode gradients.
    SSIM compares nonnegative pixels, so model outputs are mapped to [0, 1].
    This does not establish anonymity or resistance to membership inference.
    """

    def __init__(self, reference_batch, threshold=0.9):
        super().__init__()
        self.reference_batch = tf.Variable(reference_batch, trainable=False)
        self.threshold = threshold

    def calculate_privacy_loss(self, generated_batch):
        reference = (tf.cast(self.reference_batch, tf.float32) + 1.0) / 2.0
        generated = (tf.cast(generated_batch, tf.float32) + 1.0) / 2.0

        def max_similarity(image):
            return tf.reduce_max(tf.image.ssim(image[None, ...], reference, max_val=1.0))

        similarities = tf.map_fn(max_similarity, generated, fn_output_signature=tf.float32)
        return tf.reduce_mean(tf.nn.relu(similarities - self.threshold))


def save_grid(images, path):
    """Save up to 16 model outputs, with no image enhancement."""
    images = np.clip((np.asarray(images) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    images = images[:16, :, :, 0]
    size = images.shape[1]
    canvas = Image.new("L", (4 * size, math.ceil(len(images) / 4) * size))
    for index, image in enumerate(images):
        canvas.paste(Image.fromarray(image), ((index % 4) * size, (index // 4) * size))
    canvas.save(path)


class GANTrainer:
    def __init__(self, config, reference_batch, run_dir):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.generator = build_generator(config)
        self.discriminator = build_discriminator(config)
        self.gen_optimizer = tf.keras.optimizers.Adam(config.learning_rate, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(config.learning_rate, beta_1=0.5)
        # Materialize optimizer slots before validating a checkpoint restore.
        self.gen_optimizer.build(self.generator.trainable_variables)
        self.disc_optimizer.build(self.discriminator.trainable_variables)
        self.guardian = PrivacyGuardian(reference_batch, config.similarity_threshold)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.rng = tf.random.Generator.from_seed(config.seed)
        self.fixed_noise = tf.random.stateless_normal([16, config.noise_dim], seed=[config.seed, 0])
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            guardian=self.guardian,
            epoch=self.epoch,
            rng=self.rng,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, str(self.run_dir / "checkpoints"), max_to_keep=3
        )

    def restore(self):
        if not self.manager.latest_checkpoint:
            raise ValueError("No checkpoint found in this run directory")
        self.checkpoint.restore(self.manager.latest_checkpoint).assert_consumed()

    @tf.function(reduce_retracing=True)
    def train_step(self, real_images):
        noise = self.rng.normal([tf.shape(real_images)[0], self.config.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = self.generator(noise, training=True)
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(generated, training=True)
            adversarial = self.loss_fn(tf.ones_like(fake_logits), fake_logits)
            disc_loss = self.loss_fn(tf.ones_like(real_logits), real_logits) + self.loss_fn(
                tf.zeros_like(fake_logits), fake_logits
            )
            penalty = self.guardian.calculate_privacy_loss(generated)
            gen_loss = adversarial + self.config.privacy_weight * penalty
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        for gradient in gen_gradients + disc_gradients:
            tf.debugging.assert_all_finite(gradient, "Non-finite training gradient")
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        return gen_loss, disc_loss, penalty

    def train(self, dataset, epochs, steps_per_epoch=None):
        metrics_path = self.run_dir / "metrics.csv"
        write_header = not metrics_path.exists()
        with metrics_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        "epoch",
                        "generator_loss",
                        "discriminator_loss",
                        "similarity_penalty",
                        "seconds",
                        "steps",
                    ]
                )
            for _ in range(epochs):
                start = time.monotonic()
                averages = [tf.keras.metrics.Mean() for _ in range(3)]
                steps = 0
                batches = dataset.take(steps_per_epoch) if steps_per_epoch else dataset
                for batch in batches:
                    for metric, value in zip(averages, self.train_step(batch)):
                        metric.update_state(value, sample_weight=tf.shape(batch)[0])
                    steps += 1
                if not steps:
                    raise ValueError("Training dataset is empty")
                self.epoch.assign_add(1)
                epoch = int(self.epoch.numpy())
                values = [float(metric.result()) for metric in averages]
                save_grid(
                    self.generator(self.fixed_noise, training=False),
                    self.run_dir / f"epoch-{epoch:04d}.png",
                )
                self.manager.save(checkpoint_number=epoch)
                writer.writerow([epoch, *values, round(time.monotonic() - start, 3), steps])
                handle.flush()
                print(
                    f"Epoch {epoch}: generator={values[0]:.4f} "
                    f"discriminator={values[1]:.4f} similarity_penalty={values[2]:.6f} "
                    f"steps={steps}",
                    flush=True,
                )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-dir", type=Path, help="Training class folder containing JPEG/PNG images"
    )
    source.add_argument(
        "--smoke-test", action="store_true", help="Use tiny random fixtures; no medical data"
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"))
    parser.add_argument(
        "--epochs", type=int, default=1, help="Additional epochs to train (default: 1)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume a matching run and optimizer state"
    )
    parser.add_argument("--cpu", action="store_true", help="Disable GPU use")
    parser.add_argument("--max-images", type=int, help="Limit training inputs for a quick check")
    parser.add_argument("--steps-per-epoch", type=int)
    for name, field in Config.__dataclass_fields__.items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(field.default), default=field.default
        )
    args = parser.parse_args(argv)
    for name in ("epochs", "max_images", "steps_per_epoch"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    values = {name: getattr(args, name) for name in Config.__dataclass_fields__}
    if args.smoke_test:
        values.update(image_size=16, batch_size=2, gen_filters=32, disc_filters=4, reference_size=2)
    config = Config(**values)
    tf.keras.utils.set_random_seed(config.seed)
    run_metadata = {
        "config": asdict(config),
        "smoke_test": args.smoke_test,
        "data_dir": str(args.data_dir.resolve()) if args.data_dir else None,
        "max_images": args.max_images,
        "steps_per_epoch": args.steps_per_epoch,
    }
    metadata_path = args.run_dir / "config.json"
    if args.resume:
        if not metadata_path.exists():
            raise ValueError("--resume requires an existing run with config.json")
        if json.loads(metadata_path.read_text()) != run_metadata:
            raise ValueError("Resume options must match the saved configuration and data source")
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("Run directory is not empty. Use a new --run-dir or --resume")

    if args.smoke_test:
        fixtures = tf.random.stateless_uniform(
            [4, config.image_size, config.image_size, 1], seed=[config.seed, 1], minval=-1, maxval=1
        )
        dataset, count = tf.data.Dataset.from_tensor_slices(fixtures), 4
        print("Smoke test: random fixtures verify execution, not image quality.")
    else:
        dataset, count = load_dataset(args.data_dir, config, args.max_images)
    reference = next(iter(dataset.take(config.reference_size).batch(config.reference_size)))
    trainer = GANTrainer(config, reference, args.run_dir)
    if args.resume:
        trainer.restore()
    else:
        metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n")
        environment = {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "training_images": count,
            "devices": [device.name for device in tf.config.get_visible_devices()],
        }
        (args.run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    batches = dataset.shuffle(min(count, 1000), seed=config.seed).batch(config.batch_size)
    trainer.train(batches.prefetch(tf.data.AUTOTUNE), args.epochs, args.steps_per_epoch)
    print(f"Saved samples, metrics, and checkpoints to {args.run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, tf.errors.OpError) as error:
        raise SystemExit(str(error)) from error
