import json
from dataclasses import asdict

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from generate import load_generator, main, parse_args
from train import Config, main as train_main


def test_generation_restores_checkpoint_and_exports_reusable_model(tmp_path):
    run_dir = tmp_path / "training"
    train_main(["--smoke-test", "--cpu", "--run-dir", str(run_dir)])
    # A generation run needs neither the source images nor the training environment log.
    (run_dir / "environment.json").unlink()
    output_dir = tmp_path / "preview"
    args = ["--run-dir", str(run_dir), "--output-dir", str(output_dir), "--cpu"]
    main([*args, "--export-model"])
    np.testing.assert_array_equal(
        Image.open(output_dir / "samples.png"), Image.open(run_dir / "epoch-0001.png")
    )
    metadata = json.loads((output_dir / "generation.json").read_text())
    assert metadata["epoch"] == 1
    assert metadata["smoke_test"] is True
    assert metadata["count"] == 16
    model = tf.keras.models.load_model(output_dir / "generator.keras", compile=False)
    generator, config, _, _, _ = load_generator(run_dir)
    noise = tf.random.stateless_normal([3, config.noise_dim], seed=[22, 0])
    np.testing.assert_allclose(model(noise, training=False), generator(noise, training=False))
    assert model.name == "generator"
    assert not any("reference" in weight.path for weight in model.weights)
    with pytest.raises(ValueError, match="not empty"):
        main(args)

    # Inference leaves the training noise stream and checkpoint files unchanged.
    checkpoint_path = tf.train.latest_checkpoint(str(run_dir / "checkpoints"))
    rng_before = tf.train.load_variable(
        checkpoint_path, "rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE"
    )
    main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(tmp_path / "alternate"),
            "--count",
            "5",
            "--seed",
            "22",
            "--cpu",
        ]
    )
    assert Image.open(tmp_path / "alternate" / "samples.png").size == (64, 32)
    np.testing.assert_array_equal(
        rng_before,
        tf.train.load_variable(checkpoint_path, "rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE"),
    )


def test_generation_rejects_missing_checkpoint_without_creating_outputs(tmp_path):
    run_dir = tmp_path / "missing"
    output_dir = tmp_path / "preview"
    with pytest.raises(ValueError, match="config.json"):
        main(["--run-dir", str(run_dir), "--output-dir", str(output_dir)])
    assert not output_dir.exists()

    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"config": asdict(Config()), "smoke_test": False})
    )
    with pytest.raises(ValueError, match="No checkpoint"):
        main(["--run-dir", str(run_dir), "--output-dir", str(output_dir)])
    assert not output_dir.exists()


def test_generation_rejects_checkpoint_without_generator_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"config": asdict(Config()), "smoke_test": False})
    )
    epoch_only = tf.train.Checkpoint(epoch=tf.Variable(1, dtype=tf.int64, trainable=False))
    manager = tf.train.CheckpointManager(epoch_only, str(tmp_path / "checkpoints"), max_to_keep=1)
    manager.save()
    with pytest.raises(ValueError, match="does not match"):
        main(["--run-dir", str(tmp_path), "--output-dir", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("count", [0, 17])
def test_generation_rejects_invalid_grid_size(count):
    with pytest.raises(SystemExit):
        parse_args(["--run-dir", "run", "--output-dir", "preview", "--count", str(count)])
