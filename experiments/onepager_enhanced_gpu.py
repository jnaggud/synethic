#!/usr/bin/env python3
# SynthEthic: GPU-Optimized Synthetic Medical Image Generation
# Enhanced version with GPU acceleration for Mac

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import requests
import zipfile
import glob
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy import stats

# Configure TensorFlow to use GPU
try:
    # Enable mixed precision training for better performance on GPU
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # Configure GPU memory growth to prevent OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
except Exception as e:
    print(f"Warning: Could not configure GPU. Training will use CPU. Error: {e}")

# --- 1. MODEL ARCHITECTURES ---
class Generator(tf.keras.Model):
    """Generator model that creates synthetic images from random noise."""
    def __init__(self):
        super(Generator, self).__init__()
        self.model = tf.keras.Sequential([
            # Input: Random noise
            layers.Input(shape=(100,)),
            # Start with a dense layer to project noise into a starting feature map
            layers.Dense(4 * 4 * 1024, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Reshape((4, 4, 1024)),

            # Upsampling block 1: 4x4 -> 8x8
            layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            # Upsampling block 2: 8x8 -> 16x16
            layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            # Upsampling block 3: 16x16 -> 32x32
            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            # Final block to get to 64x64 and 1 channel (grayscale)
            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                 activation='tanh', use_bias=False)
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


class Discriminator(tf.keras.Model):
    """Discriminator model that classifies images as real or fake."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            # Input is a 64x64x1 image
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                         input_shape=[64, 64, 1]),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),

            layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),

            layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(1)  # No activation, using logits
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


# --- 2. CONFIGURATION ---
class Config:
    # Model hyperparameters - match original onepager.py
    IMAGE_SIZE = 64           # Size of generated images (height and width)
    BATCH_SIZE = 128          # Number of images per batch (increased from 64)
    NOISE_DIM = 100           # Size of the noise vector for generator input (matches original)
    EPOCHS = 50               # Total number of training epochs (matches original)
    LEARNING_RATE = 0.0002    # Learning rate for both generator and discriminator
    BETA_1 = 0.5              # Beta1 parameter for Adam optimizer (matches original)

    # Privacy parameters
    SIMILARITY_THRESHOLD = 0.9  # SSIM threshold for privacy checks (matches original)
    PRIVACY_LOSS_WEIGHT = 0.1   # Weight for the privacy loss term (matches original)

    # Training parameters
    SAVE_EVERY_N_EPOCHS = 5     # Save checkpoints every N epochs
    GENERATE_NUM_IMAGES = 16     # Number of images to generate for visualization

    # Directories
    CHECKPOINT_DIR = 'training_checkpoints'  # Directory for saving model checkpoints
    OUTPUT_DIR = 'poc_images'                # Directory for saving generated images

    # Dataset
    DATASET_URL = "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
    DATASET_DIR = "chest_xray_data"  # Matches original directory name

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "generated_images"), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "reports"), exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)

# --- 3. DATA LOADING & PREPROCESSING ---
class DataLoader:
    """Handles loading and preprocessing of the dataset."""

    def __init__(self):
        self.image_size = Config.IMAGE_SIZE
        self.batch_size = Config.BATCH_SIZE

    def download_and_prepare_dataset(self):
        """Download and prepare the dataset for training."""
        # Create output directory if it doesn't exist
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Download and extract dataset if not already done
        zip_path = os.path.join(Config.OUTPUT_DIR, "chest_xray.zip")
        data_dir = os.path.join(Config.OUTPUT_DIR, "chest_xray")

        if not os.path.exists(data_dir):
            print("Downloading dataset...")
            response = requests.get(Config.DATASET_URL, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(Config.OUTPUT_DIR)

            # Clean up zip file
            os.remove(zip_path)

        # Load and preprocess the dataset
        print("Loading and preprocessing dataset...")
        image_paths = glob.glob(os.path.join(data_dir, '**/*.jpeg'), recursive=True)

        # Filter for pneumonia images (as in original script)
        pneumonia_paths = [p for p in image_paths if 'PNEUMONIA' in p]

        # Limit dataset size for POC (optional)
        if len(pneumonia_paths) > 1000:
            pneumonia_paths = pneumonia_paths[:1000]

        # Load and preprocess images
        images = []
        for img_path in tqdm(pneumonia_paths, desc="Loading images"):
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((self.image_size, self.image_size))
                img_array = np.array(img)

                # Normalize to [-1, 1]
                img_array = (img_array.astype('float32') - 127.5) / 127.5

                # Add channel dimension
                img_array = np.expand_dims(img_array, axis=-1)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(np.array(images))

        # Batch and prefetch for better performance
        dataset = dataset.shuffle(buffer_size=len(images)).batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# --- 4. PRIVACY GUARDIAN ---
class PrivacyGuardian:
    """Ensures generated images don't match training data too closely."""

    def __init__(self, training_data_sample, similarity_threshold=0.9):
        self.reference_batch = training_data_sample
        self.similarity_threshold = similarity_threshold
        print("Privacy Guardian Initialized.")

    def calculate_privacy_loss(self, generated_batch):
        """Calculate privacy loss based on similarity to training data."""
        privacy_loss = 0.0
        for gen_img in generated_batch:
            max_similarity = 0
            for real_img in self.reference_batch:
                # Calculate SSIM between generated and real image
                similarity = ssim(
                    gen_img.numpy().squeeze(),
                    real_img.numpy().squeeze(),
                    data_range=2.0,
                    win_size=3
                )
                max_similarity = max(max_similarity, similarity)

            # Penalize if too similar to any training image
            if max_similarity > self.similarity_threshold:
                privacy_loss += (max_similarity - self.similarity_threshold)

        return privacy_loss / len(generated_batch)


class GANTrainer:
    """Handles GAN training with GPU optimization."""

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(
            learning_rate=Config.LEARNING_RATE,
            beta_1=Config.BETA_1
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            learning_rate=Config.LEARNING_RATE,
            beta_1=Config.BETA_1
        )

        # Setup checkpointing
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=Config.CHECKPOINT_DIR,
            max_to_keep=3
        )

        # Initialize Privacy Guardian with a sample from the dataset
        self.privacy_guardian = None

    def discriminator_loss(self, real_output, fake_output):
        """Calculate discriminator loss."""
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        """Calculate generator loss."""
        # Standard GAN loss
        gan_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        # Add privacy loss if privacy guardian is initialized
        if self.privacy_guardian is not None:
            privacy_loss = self.privacy_guardian.calculate_privacy_loss(
                self.generator(tf.random.normal([tf.shape(fake_output)[0], Config.NOISE_DIM]), training=True)
            )
            return gan_loss + Config.PRIVACY_LOSS_WEIGHT * privacy_loss
        return gan_loss

    @tf.function
    def train_step(self, real_images):
        """Single training step with GPU optimization."""
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, Config.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)

            # Discriminator forward pass
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            # Add privacy loss if privacy_guardian is initialized
            if self.privacy_guardian is not None:
                privacy_loss = self.privacy_guardian.calculate_privacy_loss(generated_images)
                gen_loss += Config.PRIVACY_LOSS_WEIGHT * privacy_loss

        # Calculate gradients and update weights
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def train(self, dataset, epochs):
        """Train the GAN with GPU optimization."""
        # Restore from checkpoint if available
        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored from {self.manager.latest_checkpoint}")

        # Get a reference batch for the Privacy Guardian
        reference_batch = next(iter(dataset.take(1)))
        self.privacy_guardian = PrivacyGuardian(reference_batch)

        # Generate fixed noise for consistent visualization
        fixed_noise = tf.random.normal([16, Config.NOISE_DIM])

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()

            # Initialize metrics
            gen_loss_avg = tf.keras.metrics.Mean()
            disc_loss_avg = tf.keras.metrics.Mean()

            # Train on batches
            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch)

                # Update metrics
                gen_loss_avg.update_state(gen_loss)
                disc_loss_avg.update_state(disc_loss)

            # Generate and save images every epoch
            self.generate_and_save_images(epoch + 1, fixed_noise)

            # Save checkpoint every N epochs
            if (epoch + 1) % Config.SAVE_EVERY_N_EPOCHS == 0:
                self.manager.save()

            # Log metrics
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Time: {time.time() - start_time:.2f}s")
            print(f"Generator Loss: {gen_loss_avg.result():.4f}")
            print(f"Discriminator Loss: {disc_loss_avg.result():.4f}")

    def generate_and_save_images(self, epoch, test_input):
        """Generate and save images for visualization.

        Args:
            epoch: Current epoch number
            test_input: Batch of noise vectors for generating sample images
        """
        # Ensure output directory exists
        if not os.path.exists('poc_images'):
            os.makedirs('poc_images')

        # Generate images from noise
        predictions = self.generator(test_input, training=False)

        # Create figure with same settings as original
        fig = plt.figure(figsize=(6, 6))

        # Display generated images in a grid
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            # Denormalize from [-1, 1] to [0, 255] and display
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        # Save with same naming convention as original
        plt.savefig(f'poc_images/image_at_epoch_{epoch:04d}.png')
        plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    # Initialize trainer
    trainer = GANTrainer(generator, discriminator)

    # Load and prepare dataset
    data_loader = DataLoader()
    dataset = data_loader.download_and_prepare_dataset()

    # Start training
    trainer.train(dataset, Config.EPOCHS)
