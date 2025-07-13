"""
This script builds a classification model using ResNet152V2, and trains it in order to identify if the audio sample is Sardinian or not.
"""

import os
import sys
import random
import librosa

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load the env variables
load_dotenv()

# load the data directory from the environment variable
DATA_DIR = os.getenv('DATA_DIR')
if DATA_DIR is None:
    sys.exit("ERROR: DATA_DIR environment variable not set in .env file.")

# Configuring and defining the constants
CONFIG = {
    "SR": 24000,
    "N_MELS": 128,
    "WIN_LENGTH": 1024,
    "HOP_LENGTH": 512,
    "TARGET_SIZE": (224, 224),
    "BATCH_SIZE": 16,
    "NUM_EPOCHS": 5,
    "SEED": 42,
    "METADATA_PATH": os.path.join(DATA_DIR, "metadata.csv"),
    "AUDIO_BASE_PATH": DATA_DIR,
    "CHECKPOINT_PATH": "srdorigin_models/best_model.keras",
    "LOG_PATH": "training_log.csv"
}

# setting random seed
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
tf.random.set_seed(CONFIG["SEED"])

def preprocess_audio_to_image(file_path, 
                              sr=24000, 
                              n_mels=128,
                              win_length=1024, 
                              hop_length=512, 
                              target_size=(224, 224),
                              eps=1e-6):
    """Converts an audio file into a resized mel-spectrogram image for model input."""
    
    audio, _ = librosa.load(file_path, sr=CONFIG["SR"])

    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=CONFIG["SR"], 
        n_fft=CONFIG["WIN_LENGTH"],
        hop_length=CONFIG["HOP_LENGTH"], 
        n_mels=CONFIG["N_MELS"]
    )

    mel_spec = np.maximum(mel_spec, eps)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)

    finite_vals = log_mel_spec[np.isfinite(log_mel_spec)]
    if finite_vals.size:
        floor = finite_vals.min()
        log_mel_spec = np.where(np.isfinite(log_mel_spec), log_mel_spec, floor)
    else:
        log_mel_spec = np.zeros_like(log_mel_spec)

    min_val, max_val = log_mel_spec.min(), log_mel_spec.max()
    denom = max_val - min_val
    
    if denom > 0:
        log_mel_spec_norm = (log_mel_spec - min_val) / denom
    else:
        log_mel_spec_norm = np.zeros_like(log_mel_spec)

    img = np.stack([log_mel_spec_norm]*3, axis=-1)
    img_tensor = tf.image.resize(img, CONFIG["TARGET_SIZE"])
    
    return img_tensor.numpy()


def create_dataset(file_paths, labels, batch_size=16, shuffle=True):
    """Creates a TensorFlow dataset from file paths and labels."""
    def _load_and_preprocess(path, label):
        path = path.numpy().decode('utf-8')
        img = preprocess_audio_to_image(path)

        return img, np.int32(label)

    def _wrapper(path, label):
        img, label = tf.py_function(_load_and_preprocess, [path, label], [tf.float32, tf.int32])
        img.set_shape((CONFIG["TARGET_SIZE"][0], CONFIG["TARGET_SIZE"][1], 3))
        label.set_shape(())
        
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=CONFIG["SEED"])
    
    dataset = dataset.map(_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_model(input_shape=(224, 224, 3), fine_tune=False, num_trainable_layers=10):
    """Builds a ResNet152V2 model for binary classification."""
    
    base_model = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet',
                                                   input_shape=input_shape)
    
    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:-num_trainable_layers]:
            layer.trainable = False
    else:
        base_model.trainable = True

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=not fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def full_path(file_name):
    return os.path.join(CONFIG["AUDIO_BASE_PATH"], file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ResNet model to classify Sardinian audio samples.")
    parser.add_argument("--epochs", type=int, default=CONFIG["NUM_EPOCHS"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"], help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=CONFIG["CHECKPOINT_PATH"], help="Path to save best model")
    parser.add_argument("--log_path", type=str, default=CONFIG["LOG_PATH"], help="Path to save training log")
    parser.add_argument("--fine_tune", action="store_true", help="Enable fine-tuning of base model")
    parser.add_argument("--num_trainable_layers", type=int, default=10, help="Number of trainable layers if fine-tuning")
    args = parser.parse_args()

    CONFIG["NUM_EPOCHS"] = args.epochs
    CONFIG["BATCH_SIZE"] = args.batch_size
    CONFIG["CHECKPOINT_PATH"] = args.checkpoint
    CONFIG["LOG_PATH"] = args.log_path
    
    metadata_df = pd.read_csv(CONFIG["METADATA_PATH"], sep="|")
    metadata_df.columns = ['audio_file', 'srdorg']
    metadata_df = metadata_df.sample(frac=1, random_state=CONFIG["SEED"]).reset_index(drop=True)
    metadata_df['audio_file'] = metadata_df['audio_file'].apply(full_path)
    
    file_paths = metadata_df['audio_file'].tolist()
    labels = metadata_df['srdorg'].tolist()

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=CONFIG["SEED"])
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_paths, val_labels, test_size=0.5, random_state=CONFIG["SEED"])

    train_ds = create_dataset(train_paths, train_labels, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_ds = create_dataset(val_paths, val_labels, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    test_ds = create_dataset(test_paths, test_labels, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    model = build_model(fine_tune=args.fine_tune, num_trainable_layers=args.num_trainable_layers)

    callbacks = [
        tf.keras.callbacks.CSVLogger(CONFIG["LOG_PATH"], separator=',', append=False),
        tf.keras.callbacks.ModelCheckpoint(CONFIG["CHECKPOINT_PATH"], monitor='val_loss',
                                           save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    history = model.fit(train_ds,
                        epochs=CONFIG["NUM_EPOCHS"],
                        validation_data=val_ds,
                        callbacks=callbacks)