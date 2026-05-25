"""
Train an InceptionV3-based waste classifier.

Requirements:
  - kaggle.json in ~/.kaggle/ (for dataset download)
  - pip install kaggle tensorflow

Usage:
  python scripts/train_inception.py
"""
import os
import sys
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "garbage_classification"

CLASSES = ["Cardboard", "Trash", "Plastic", "Metal", "Glass", "Paper"]
IMG_SIZE = (384, 512)
BATCH = 32
EPOCHS = 15


def download_dataset():
    if DATA_DIR.exists():
        print(f"Dataset already at {DATA_DIR}")
        return
    try:
        import kaggle
    except ImportError:
        sys.exit("Install kaggle: pip install kaggle")
    print("Downloading dataset from Kaggle...")
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    os.system(
        f'kaggle datasets download -d asdasdasasdas/garbage-classification '
        f'-p "{DATA_DIR.parent}" --unzip'
    )
    print("Download complete")


def train():
    import tensorflow as tf
    from tensorflow.keras import layers

    download_dataset()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="nearest",
    )
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="training",
        classes=CLASSES,
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="validation",
        classes=CLASSES,
    )

    base = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Training for {EPOCHS} epochs...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    MODELS_DIR.mkdir(exist_ok=True)
    out = str(MODELS_DIR / "inception_model.keras")
    model.save(out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    train()
