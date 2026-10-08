import csv

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from train import Config, GANTrainer, PrivacyGuardian, load_dataset, main


def tiny_config():
    return Config(
        image_size=16, batch_size=2, noise_dim=8, gen_filters=32, disc_filters=4, reference_size=2
    )


def test_data_loading_keeps_splits_separate_and_handles_png(tmp_path):
    train_dir = tmp_path / "train" / "PNEUMONIA"
    test_dir = tmp_path / "test" / "PNEUMONIA"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    Image.new("L", (24, 24), 0).save(train_dir / "a.png")
    Image.new("RGB", (24, 24), (255, 255, 255)).save(train_dir / "b.jpeg")
    Image.new("L", (24, 24), 128).save(test_dir / "held-out.png")
    with pytest.raises(ValueError, match="No JPEG/PNG"):
        load_dataset(tmp_path, tiny_config())
    dataset, count = load_dataset(train_dir, tiny_config())
    assert count == 2
    batch = next(iter(dataset.batch(2))).numpy()
    assert batch.shape == (2, 16, 16, 1)
    np.testing.assert_allclose(batch[0], -1)
    np.testing.assert_allclose(batch[1], 1)


def test_similarity_penalty_has_gradients_in_graph_mode():
    reference = tf.random.stateless_uniform([2, 16, 16, 1], [7, 1], minval=-1, maxval=1)
    guardian = PrivacyGuardian(reference, threshold=0.5)

    @tf.function
    def penalty_and_gradient(images):
        with tf.GradientTape() as tape:
            tape.watch(images)
            loss = guardian.calculate_privacy_loss(images)
        return loss, tape.gradient(loss, images)

    loss, gradient = penalty_and_gradient(reference * 0.8)
    assert float(loss) > 0
    assert gradient is not None
    assert np.isfinite(gradient).all()
    assert np.any(np.abs(gradient.numpy()) > 0)
    np.testing.assert_allclose(guardian.calculate_privacy_loss(reference), 0.5, atol=1e-5)
    assert float(PrivacyGuardian(reference, threshold=1).calculate_privacy_loss(reference)) == 0


def test_training_updates_weights_and_restores_full_checkpoint(tmp_path):
    config = tiny_config()
    reference = tf.random.stateless_uniform([2, 16, 16, 1], [3, 2], minval=-1, maxval=1)
    trainer = GANTrainer(config, reference, tmp_path)
    before = trainer.generator.trainable_variables[0].numpy().copy()
    trainer.train(tf.data.Dataset.from_tensor_slices(reference).batch(2), epochs=1)
    assert not np.array_equal(before, trainer.generator.trainable_variables[0].numpy())
    expected = trainer.generator(trainer.fixed_noise, training=False).numpy()
    restored = GANTrainer(config, tf.zeros_like(reference), tmp_path)
    restored.restore()
    np.testing.assert_allclose(restored.generator(restored.fixed_noise, training=False), expected)
    np.testing.assert_array_equal(restored.guardian.reference_batch, reference)
    assert int(restored.epoch) == 1
    assert int(restored.gen_optimizer.iterations) == 1
    assert int(restored.disc_optimizer.iterations) == 1
    np.testing.assert_array_equal(restored.rng.state, trainer.rng.state)
    assert all(np.isfinite(value) for value in restored.train_step(reference[:1]))
    assert Image.open(tmp_path / "epoch-0001.png").size == (64, 64)


def test_cli_smoke_resume_and_overwrite_protection(tmp_path):
    args = ["--smoke-test", "--cpu", "--run-dir", str(tmp_path / "run")]
    main(args)
    with pytest.raises(ValueError, match="not empty"):
        main(args)
    main([*args, "--resume"])
    with (tmp_path / "run" / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    with pytest.raises(ValueError, match="must match"):
        main([*args, "--resume", "--seed", "99"])


@pytest.mark.parametrize(
    "values",
    [
        {"image_size": 24},
        {"batch_size": 0},
        {"gen_filters": 12},
        {"privacy_weight": -1},
        {"similarity_threshold": 1.1},
        {"learning_rate": float("nan")},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValueError):
        Config(**values)
