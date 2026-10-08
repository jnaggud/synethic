# SynEthic: archived DCGAN prototype. Use ../train.py for the supported trainer.
# This script builds and trains a DCGAN to generate synthetic chest X-ray images.
# The experimental similarity penalty does not provide a privacy guarantee.

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import requests
import zipfile
import glob
from PIL import Image

# --- 1. CONFIGURATION & SETUP ---
# Fixed hyperparameters used by this prototype.
IMAGE_SIZE = 64  # For faster training, we'll use 64x64. For higher fidelity, this would be increased.
BATCH_SIZE = 128
NOISE_DIM = 100  # The dimension of the random noise vector fed to the generator.
EPOCHS = 500      # Number of times we train on the entire dataset. 50 is a good starting point.
LEARNING_RATE = 0.0002
BETA_1 = 0.5     # Parameter for the Adam optimizer, recommended for GANs.

# --- 2. DATA LOADING & PREPARATION
# ---
# Download the public dataset and preprocess the PNEUMONIA training class.

def download_and_prepare_dataset():
    """
    Downloads the chest X-ray dataset from Mendeley and prepares it for training.
    """
    dataset_url = "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
    zip_path = "chest_xray.zip"
    data_dir = "chest_xray_data"

    if not os.path.exists(data_dir):
        print("Downloading dataset...")
        with open(zip_path, "wb") as f:
            r = requests.get(dataset_url)
            f.write(r.content)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    # We will use only the pneumonia images for this POC
    image_paths = glob.glob(os.path.join(data_dir, 'chest_xray', 'train', 'PNEUMONIA', '*.jpeg'))

    print(f"Found {len(image_paths)} pneumonia images.")

    # Create a tf.data.Dataset for efficient processing
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=1) # Read as grayscale
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        # Normalize images to [-1, 1]. This is crucial for GAN training stability.
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5
        return image

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(len(image_paths)).batch(BATCH_SIZE)

    return dataset

# --- 3. THE GENERATOR (The "Artist") ---
# Architectural Component: This network takes random noise and transforms it into a synthetic image.
# It uses Conv2DTranspose layers to "upsample" the noise into a 64x64 grayscale image.
def build_generator():
    model = models.Sequential(name="Generator")

    # Start with a dense layer to project noise into a starting feature map
    model.add(layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 1024)))

    # Upsampling block 1: 4x4 -> 8x8
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsampling block 2: 8x8 -> 16x16
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsampling block 3: 16x16 -> 32x32
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Final block to get to 64x64 and 1 channel (grayscale)
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# --- 4. THE DISCRIMINATOR (The "Critic") ---
# Architectural Component: This network is a standard convolutional classifier.
# It takes an image and outputs a single value indicating whether it thinks the image is "Real" or "Fake".
def build_discriminator():
    model = models.Sequential(name="Discriminator")

    # Input is a 64x64x1 image
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=[IMAGE_SIZE, IMAGE_SIZE, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # No activation function here, we output raw logits

    return model

# --- 5. EXPERIMENTAL REFERENCE-SIMILARITY PENALTY ---
# This legacy implementation uses signed pixels; see ../train.py for corrected SSIM scaling.
class PrivacyGuardian:
    def __init__(self, training_data_sample, similarity_threshold=0.9):
        """
        Initializes the guardian with a sample of real data to check against.
        Args:
            training_data_sample (Tensor): A batch of real images to serve as a reference.
            similarity_threshold (float): The threshold above which a generated image is considered "too similar".
        """
        print("Privacy Guardian Initialized.")
        self.reference_batch = training_data_sample
        # Compare local image structure using SSIM.
        self.similarity_threshold = similarity_threshold

    def calculate_privacy_loss(self, generated_batch):
        """
        Calculates a loss term that penalizes the generator for creating images that are too similar
        to the real training data.
        """
        max_similarities = []
        # For each generated image, find its max similarity to any image in the reference batch
        for i in range(generated_batch.shape[0]):
            g_img = tf.expand_dims(generated_batch[i], 0)
            # Legacy range handling is retained here; it is corrected in the supported trainer.
            ssim_scores = tf.image.ssim(g_img, self.reference_batch, max_val=2.0)
            max_similarity = tf.reduce_max(ssim_scores)
            max_similarities.append(max_similarity)

        # Calculate the "privacy penalty" for images that exceed the threshold
        privacy_violations = tf.maximum(0.0, tf.stack(max_similarities) - self.similarity_threshold)
        privacy_loss = tf.reduce_mean(privacy_violations)

        # Historical constant offset; it does not change the gradient.
        return privacy_loss + 1e-6

# --- 6. LOSS FUNCTIONS & OPTIMIZERS ---
# Binary cross-entropy objectives for adversarial training.
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy_loss(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # The generator wants the discriminator to think its fake images are real.
    return cross_entropy_loss(tf.ones_like(fake_output), fake_output)

generator_optimizer = optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)


# --- 7. THE TRAINING LOOP ---
# Train the models and save fixed-noise samples at epoch boundaries.

# A decorator to compile a function into a high-performance TensorFlow graph.
@tf.function
def train_step(images, generator, discriminator, guardian, privacy_loss_weight=0.1):
    # Generate a batch of random noise
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # Get discriminator's predictions for real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate standard GAN losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # Add the experimental similarity term.
        # Calculate the privacy loss and add it to the generator's loss
        privacy_loss = guardian.calculate_privacy_loss(generated_images)
        total_gen_loss = gen_loss + (privacy_loss_weight * privacy_loss)
        # ------------------------------------

    # Calculate gradients and apply them to update the models' weights
    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, privacy_loss

def generate_and_save_images(model, epoch, test_input):
    """Helper function to save a grid of generated images."""
    if not os.path.exists('poc_images'):
        os.makedirs('poc_images')

    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(6, 6))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # Denormalize image from [-1, 1] to [0, 255] and display
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('poc_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)

def train(dataset, epochs, generator, discriminator):
    print("\nStarting Training...")
    # We will use a fixed noise vector to see how the generator improves over time on the same input.
    seed = tf.random.normal([16, NOISE_DIM])

    # Get a reference batch for the Privacy Guardian
    reference_batch = next(iter(dataset))
    guardian = PrivacyGuardian(reference_batch)

    for epoch in range(epochs):
        start = time.time()
        gen_loss_total = 0
        disc_loss_total = 0
        privacy_loss_total = 0
        num_batches = 0

        for image_batch in dataset:
            # The last batch may not be full size
            if image_batch.shape[0] != BATCH_SIZE:
                continue

            g_loss, d_loss, p_loss = train_step(image_batch, generator, discriminator, guardian)
            gen_loss_total += g_loss
            disc_loss_total += d_loss
            privacy_loss_total += p_loss
            num_batches += 1

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed)

        avg_gen_loss = gen_loss_total / num_batches
        avg_disc_loss = disc_loss_total / num_batches
        avg_privacy_loss = privacy_loss_total / num_batches

        print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec')
        print(f'Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Privacy Loss: {avg_privacy_loss:.4f}')

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    print("\nTraining Complete.")


# --- 8. MAIN EXECUTION ---
if __name__ == '__main__':
    # 1. Prepare the data
    train_dataset = download_and_prepare_dataset()

    # 2. Build the models
    generator = build_generator()
    discriminator = build_discriminator()

    # Display the model architectures.
    print("\n--- Generator Architecture ---")
    generator.summary()
    print("\n--- Discriminator Architecture ---")
    discriminator.summary()

    # 3. Start the training process
    train(train_dataset, EPOCHS, generator, discriminator)

    print("\nSynthEthic POC finished.")
    print("Check the 'poc_images' directory for generated synthetic X-rays.")
