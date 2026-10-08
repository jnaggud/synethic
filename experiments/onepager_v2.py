# SynEthic: archived configurable prototype. Use ../train.py for supported training.
# This version allows changing IMAGE_SIZE and other parameters without breaking the model

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import requests
import zipfile
from tqdm import tqdm
from PIL import Image

# ===========================================
# CONFIGURATION - All parameters are here
# ===========================================
class Config:
    # Image and model parameters
    IMAGE_SIZE = 64           # Can be changed to 128 or other powers of 2
    CHANNELS = 1              # 1 for grayscale, 3 for RGB
    BATCH_SIZE = 128
    EPOCHS = 200
    SEED = 42

    # Generator parameters
    NOISE_DIM = 100
    GEN_FILTERS = 1024        # Base number of filters in generator
    GEN_KERNEL_SIZE = 5       # Kernel size for generator
    GEN_STRIDES = 2           # Stride for generator

    # Discriminator parameters
    DISC_FILTERS = 64         # Base number of filters in discriminator
    DISC_KERNEL_SIZE = 5      # Kernel size for discriminator
    DISC_STRIDES = 2          # Stride for discriminator
    DROPOUT_RATE = 0.3        # Dropout rate in discriminator

    # Training parameters
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5
    LOSS_FN = 'binary_crossentropy'

    # Privacy parameters
    PRIVACY_WEIGHT = 0.1
    SIMILARITY_THRESHOLD = 0.9

    # Paths
    DATASET_URL = "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
    DATASET_DIR = "chest_xray_data"
    OUTPUT_DIR = "generated_images"
    CHECKPOINT_DIR = "training_checkpoints"

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.DATASET_DIR, exist_ok=True)

# ===========================================
# DATA LOADING & PREPROCESSING
# ===========================================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.image_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)

    def download_and_prepare_dataset(self):
        """Download and prepare the dataset."""
        if not os.path.exists(self.config.DATASET_DIR):
            print("Downloading dataset...")
            zip_path = "chest_xray.zip"

            # Download the dataset
            response = requests.get(self.config.DATASET_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f:
                for data in tqdm(response.iter_content(1024),
                               total=total_size//1024,
                               unit='KB',
                               unit_scale=True):
                    f.write(data)

            # Extract the dataset
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.DATASET_DIR)

            # Clean up
            os.remove(zip_path)
            print("Dataset downloaded and extracted.")

        # Load and preprocess images
        image_paths = []
        for root, _, files in os.walk(self.config.DATASET_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(self._load_and_preprocess_image,
                            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=self.config.CHANNELS)
        img = tf.image.resize(img, self.image_size)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        return img

# ===========================================
# MODEL ARCHITECTURES
# ===========================================
class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        # Calculate the initial size based on the target image size
        # We need to go through 4 strided convolutions (2^4 = 16)
        self.initial_size = config.IMAGE_SIZE // 16
        self.initial_channels = config.GEN_FILTERS

        self.model = tf.keras.Sequential([
            # Input: Random noise
            layers.Input(shape=(config.NOISE_DIM,)),

            # Project and reshape
            layers.Dense(self.initial_size * self.initial_size * self.initial_channels,
                        use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((self.initial_size, self.initial_size, self.initial_channels)),

            # Upsampling blocks
            self._make_upsample_block(self.initial_channels // 2),  # 4x4 -> 8x8
            self._make_upsample_block(self.initial_channels // 4),  # 8x8 -> 16x16
            self._make_upsample_block(self.initial_channels // 8),  # 16x16 -> 32x32
            self._make_upsample_block(self.initial_channels // 16), # 32x32 -> 64x64

            # Final convolution to get to desired channels
            layers.Conv2DTranspose(
                config.CHANNELS,
                (config.GEN_KERNEL_SIZE, config.GEN_KERNEL_SIZE),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                activation='tanh'
            )
        ])

    def _make_upsample_block(self, filters):
        """Helper function to create an upsampling block."""
        return tf.keras.Sequential([
            layers.Conv2DTranspose(
                filters,
                (self.config.GEN_KERNEL_SIZE, self.config.GEN_KERNEL_SIZE),
                strides=(self.config.GEN_STRIDES, self.config.GEN_STRIDES),
                padding='same',
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.model = tf.keras.Sequential([
            # Input layer
            layers.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.CHANNELS)),

            # Downsample blocks
            self._make_downsample_block(config.DISC_FILTERS, normalize=False),    # 64x64 -> 32x32
            self._make_downsample_block(config.DISC_FILTERS * 2),                 # 32x32 -> 16x16
            self._make_downsample_block(config.DISC_FILTERS * 4),                 # 16x16 -> 8x8
            self._make_downsample_block(config.DISC_FILTERS * 8),                 # 8x8 -> 4x4

            # Final layers
            layers.Flatten(),
            layers.Dense(1)  # Using logits (no activation)
        ])

    def _make_downsample_block(self, filters, normalize=True):
        """Helper function to create a downsampling block."""
        block = tf.keras.Sequential()
        block.add(layers.Conv2D(
            filters,
            (self.config.DISC_KERNEL_SIZE, self.config.DISC_KERNEL_SIZE),
            strides=(self.config.DISC_STRIDES, self.config.DISC_STRIDES),
            padding='same'
        ))
        if normalize:
            block.add(layers.BatchNormalization())
        block.add(layers.LeakyReLU(alpha=0.2))
        block.add(layers.Dropout(self.config.DROPOUT_RATE))
        return block

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

# ===========================================
# PRIVACY GUARDIAN
# ===========================================
class PrivacyGuardian:
    def __init__(self, reference_batch, threshold=0.9):
        self.reference_batch = reference_batch
        self.threshold = threshold

    def calculate_privacy_loss(self, generated_images):
        """Calculate privacy loss based on similarity to reference images."""
        # Resize generated images to match reference batch if needed
        if generated_images.shape[1:] != self.reference_batch.shape[1:]:
            generated_images = tf.image.resize(
                generated_images,
                self.reference_batch.shape[1:3]
            )

        # Calculate SSIM for each generated image against all reference images
        max_similarities = []
        for gen_img in generated_images:
            similarities = []
            for ref_img in self.reference_batch:
                # Calculate SSIM (Structural Similarity Index)
                ssim = tf.image.ssim(
                    tf.expand_dims(gen_img, 0),
                    tf.expand_dims(ref_img, 0),
                    max_val=2.0  # Since our images are in [-1, 1]
                )
                similarities.append(ssim)

            # Get maximum similarity for this generated image
            max_similarity = tf.reduce_max(tf.concat(similarities, axis=0))
            max_similarities.append(max_similarity)

        # Calculate privacy loss
        privacy_loss = 0.0
        for sim in max_similarities:
            if sim > self.threshold:
                privacy_loss += (sim - self.threshold)

        return privacy_loss / len(max_similarities)

# ===========================================
# TRAINING LOOP
# ===========================================
class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

        # Optimizers
        self.gen_optimizer = optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            beta_1=config.BETA_1
        )
        self.disc_optimizer = optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            beta_1=config.BETA_1
        )

        # Loss function
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Checkpoints
        self.checkpoint_dir = config.CHECKPOINT_DIR
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=3
        )

        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    @tf.function
    def train_step(self, real_images):
        """Single training step."""
        batch_size = tf.shape(real_images)[0]

        # Generate noise
        noise = tf.random.normal([batch_size, self.config.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)

            # Discriminator forward pass
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            # Add privacy loss if privacy_guardian is available
            if hasattr(self, 'privacy_guardian'):
                privacy_loss = self.privacy_guardian.calculate_privacy_loss(generated_images)
                gen_loss += self.config.PRIVACY_WEIGHT * privacy_loss

        # Calculate gradients
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        # Apply gradients
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def generator_loss(self, fake_output):
        """Generator loss function."""
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss function."""
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def train(self, dataset, epochs):
        """Training loop."""
        # Restore from checkpoint if available
        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored from {self.manager.latest_checkpoint}")

        # Initialize privacy guardian with a batch from the dataset
        reference_batch = next(iter(dataset.take(1)))[:16]  # Use first 16 images as reference
        self.privacy_guardian = PrivacyGuardian(
            reference_batch,
            threshold=self.config.SIMILARITY_THRESHOLD
        )

        # Fixed noise for generating sample images
        fixed_noise = tf.random.normal([16, self.config.NOISE_DIM])

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()

            # Initialize metrics
            gen_loss_avg = tf.keras.metrics.Mean()
            disc_loss_avg = tf.keras.metrics.Mean()

            # Train on batches
            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch)
                gen_loss_avg.update_state(gen_loss)
                disc_loss_avg.update_state(disc_loss)

            # Generate and save images
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.generate_and_save_images(epoch + 1, fixed_noise)
                self.manager.save()

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Gen Loss: {gen_loss_avg.result():.4f}, "
                  f"Disc Loss: {disc_loss_avg.result():.4f}, "
                  f"Time: {time.time() - start_time:.2f}s")

    def generate_and_save_images(self, epoch, test_input):
        """Generate and save images for visualization."""
        predictions = self.generator(test_input, training=False)

        # Rescale images from [-1, 1] to [0, 1]
        predictions = (predictions + 1) / 2.0

        # Create a grid of images
        fig = plt.figure(figsize=(10, 10))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            img = predictions[i, :, :, 0]  # For grayscale
            plt.imshow(img, cmap='gray')
            plt.axis('off')

        # Save the figure
        output_path = os.path.join(
            self.config.OUTPUT_DIR,
            f'epoch_{epoch:04d}.png'
        )
        plt.savefig(output_path)
        plt.close()

# ===========================================
# MAIN EXECUTION
# ===========================================
def main():
    # Setup configuration
    config = Config()
    config.setup_directories()

    # Set random seed for reproducibility
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    # Print configuration
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    for key, value in vars(config).items():
        if not key.startswith('_') and not callable(value):
            print(f"{key}: {value}")
    print("=" * 50)

    # Load and prepare dataset
    print("\nLoading dataset...")
    data_loader = DataLoader(config)
    dataset = data_loader.download_and_prepare_dataset()

    # Initialize and train the model
    print("\nInitializing model...")
    trainer = GANTrainer(config)

    print("\nStarting training...")
    trainer.train(dataset, config.EPOCHS)
    print("Training completed!")

if __name__ == "__main__":
    main()
