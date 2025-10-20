import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

num_threads = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
print(f"Using {num_threads} CPU threads for TensorFlow.")

# ------------------------
# Config
# ------------------------
IMAGE_SIZE = (224, 224)             # resizing target
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

DATA_ROOT = Path('.')               # root containing the class folders
CLASS_NAMES = ['Painting', 'Photo', 'Schematics', 'Sketch', 'Text']
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 20
MODEL_SAVE_PATH = 'multitask_photo_model.keras'

# ------------------------
# Utility functions
# ------------------------
def gather_image_paths_and_labels(root: Path, class_names: List[str]) -> pd.DataFrame:
    """
    Return a DataFrame with columns: ['filepath', 'class_name', 'is_photo']
    """
    rows = []
    for cls in class_names:
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"Warning: directory {cls_dir} does not exist and will be skipped.")
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'):
            for p in cls_dir.glob(ext):
                rows.append({
                    'filepath': str(p.resolve()),
                    'class_name': cls,
                    'is_photo': 1 if cls == 'Photo' else 0
                })
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle
    return df

def load_and_preprocess_image_np(path: str, image_size: Tuple[int,int]=IMAGE_SIZE) -> np.ndarray:
    """
    Load an image from disk as numpy array, convert to RGB if needed,
    resize, and scale to [0,1]. This function will be called via tf.numpy_function.
    """
    try:
        img = imageio.v2.imread(path)
    except Exception as e:
        raise RuntimeError(f"Error reading image {path}: {e}")
    if img is None:
        raise RuntimeError(f"imageio returned None for {path}")
    # If grayscale, convert to 3 channels
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    # If image has alpha channel, drop it
    if img.shape[-1] == 4:
        img = img[..., :3]
    # Resize using tensorflow's backend for good interpolation; use numpy here and cast to float32
    img_tf = tf.image.resize(img, image_size, method='bilinear').numpy()
    img_tf = img_tf.astype(np.float32) / 255.0
    # In rare cases the image may have >3 channels; keep only first 3
    if img_tf.shape[-1] > 3:
        img_tf = img_tf[..., :3]
    # Ensure shape is (H, W, 3)reports
    img_tf = img_tf.reshape((image_size[0], image_size[1], 3))
    return img_tf

def build_tf_dataset(df: pd.DataFrame,
                     batch_size: int = BATCH_SIZE,
                     shuffle: bool = True,
                     augment: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that yields (image, {'binary_output': b, 'multiclass_output': c})
    where b is scalar 0/1 and c is one-hot vector length NUM_CLASSES.
    """
    paths = df['filepath'].values
    binary_labels = df['is_photo'].astype(np.float32).values
    class_labels = pd.Categorical(df['class_name'], categories=CLASS_NAMES).codes
    class_labels_onehot = tf.one_hot(class_labels, depth=NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((paths, binary_labels, class_labels_onehot))

    def _load_fn(path, binary_label, class_onehot):
        # path is a bytestring -> convert to string
        path_str = path.numpy().decode('utf-8')
        img = load_and_preprocess_image_np(path_str, IMAGE_SIZE)
        return img.astype(np.float32), np.float32(binary_label), np.float32(class_onehot)

    def _tf_map(path, binary_label, class_onehot):
        img, b, c = tf.py_function(func=_load_fn,
                                   inp=[path, binary_label, class_onehot],
                                   Tout=[tf.float32, tf.float32, tf.float32])
        img.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        b.set_shape(())
        c.set_shape((NUM_CLASSES,))
        return img, {'binary_output': b, 'multiclass_output': c}

    ds = ds.map(_tf_map, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED)

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# ------------------------
# Data augmentation
# ------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(height_factor=(-0.05, 0.0), width_factor=(-0.05, 0.0)),
], name='data_augmentation')

# ------------------------
# Model (shared backbone + two heads)
# ------------------------
def build_multitask_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    Build a convolutional neural network with a shared backbone and two heads:
    - binary_output: sigmoid (is photo)
    - multiclass_output: softmax (which of the 5 classes)
    """
    inputs = layers.Input(shape=input_shape, name='image_input')
    x = inputs

    # Shared convolutional backbone (small, can be replaced by pretrained backbone)
    x = layers.Rescaling(1.0)(x)  # images are already in [0,1], this is a placeholder
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    shared = layers.Dense(256, activation='relu')(x)

    # Binary head: is_photo
    b = layers.Dense(64, activation='relu')(shared)
    b = layers.Dropout(0.2)(b)
    binary_output = layers.Dense(1, activation='sigmoid', name='binary_output')(b)

    # Multiclass head
    m = layers.Dense(64, activation='relu')(shared)
    m = layers.Dropout(0.2)(m)
    multiclass_output = layers.Dense(num_classes, activation='softmax', name='multiclass_output')(m)

    model = models.Model(inputs=inputs, outputs=[binary_output, multiclass_output], name='multitask_cnn')
    return model

# ------------------------
# Main training pipeline
# ------------------------
def main():
    # Gather dataset filepaths and labels
    df = gather_image_paths_and_labels(DATA_ROOT, CLASS_NAMES)
    print(f"Found {len(df)} images across classes:")
    print(df['class_name'].value_counts())

    # Train/val/test split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['class_name'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['class_name'])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build datasets
    train_ds = build_tf_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_ds = build_tf_dataset(val_df, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    test_ds = build_tf_dataset(test_df, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    # Build model
    model = build_multitask_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.summary()

    # Compile model with two losses
    losses = {
        'binary_output': tf.keras.losses.BinaryCrossentropy(),
        'multiclass_output': tf.keras.losses.CategoricalCrossentropy()
    }
    loss_weights = {
        'binary_output': 1.0,
        'multiclass_output': 1.0
    }
    metrics = {
        'binary_output': [tf.keras.metrics.BinaryAccuracy(name='binary_acc'), tf.keras.metrics.AUC(name='binary_auc')],
        'multiclass_output': [tf.keras.metrics.CategoricalAccuracy(name='multiclass_acc')]
    }

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics=metrics)

    # Callbacks
    cb = [
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, save_weights_only=False),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    # Fit
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS,
                        callbacks=cb)

    # Save final model (again)
    model.save(MODEL_SAVE_PATH)

    # Plot training curves
    plot_history(history)

    # Evaluate on test set and produce detailed reports
    evaluate_model_on_test(model, test_df, test_ds)

# ------------------------
# Plot helpers
# ------------------------
def plot_history(history: tf.keras.callbacks.History):
    """
    Plot loss and accuracy curves for both heads.
    """
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 5))
    # total loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], label='train_total_loss')
    plt.plot(epochs, hist['val_loss'], label='val_total_loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # binary acc and multiclass acc
    plt.subplot(1, 2, 2)
    if 'binary_output_binary_acc' in hist:
        plt.plot(epochs, hist['binary_output_binary_acc'], label='train_binary_acc')
        plt.plot(epochs, hist['val_binary_output_binary_acc'], label='val_binary_acc')
    if 'multiclass_output_multiclass_acc' in hist:
        plt.plot(epochs, hist['multiclass_output_multiclass_acc'], label='train_multiclass_acc')
        plt.plot(epochs, hist['val_multiclass_output_multiclass_acc'], label='val_multiclass_acc')
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------
# Evaluation
# ------------------------
def evaluate_model_on_test(model: tf.keras.Model, test_df: pd.DataFrame, test_ds: tf.data.Dataset):
    """
    Run predictions on the test dataset and print classification reports for both tasks.
    """
    # Predict: iterate through test_ds to collect predictions and true labels
    binary_preds = []
    multiclass_preds = []
    binary_trues = []
    multiclass_trues = []
    filepaths = []

    # We will load files from test_df in the same order for consistent reporting
    for idx, row in test_df.reset_index(drop=True).iterrows():
        path = row['filepath']
        # load and preprocess using the same function
        img = load_and_preprocess_image_np(path, IMAGE_SIZE)
        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
        b_pred, m_pred = model.predict(img_batch, verbose=0)
        b_pred_val = float(b_pred.squeeze())
        m_pred_val = m_pred.squeeze()
        binary_preds.append(1 if b_pred_val >= 0.5 else 0)
        multiclass_preds.append(int(np.argmax(m_pred_val)))
        binary_trues.append(int(row['is_photo']))
        multiclass_trues.append(int(pd.Categorical([row['class_name']], categories=CLASS_NAMES).codes[0]))
        filepaths.append(path)

    # Binary report
    print("\n=== Binary classification report (is_photo) ===")
    print(classification_report(binary_trues, binary_preds, target_names=['not_photo', 'photo']))

    # Multiclass report
    print("\n=== Multiclass classification report ===")
    print(classification_report(multiclass_trues, multiclass_preds, target_names=CLASS_NAMES))

    # Show confusion matrix for multiclass
    cm = confusion_matrix(multiclass_trues, multiclass_preds)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Multiclass Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Save a CSV with predictions
    out_df = pd.DataFrame({
        'filepath': filepaths,
        'binary_true': binary_trues,
        'binary_pred': binary_preds,
        'multiclass_true': [CLASS_NAMES[i] for i in multiclass_trues],
        'multiclass_pred': [CLASS_NAMES[i] for i in multiclass_preds]
    })
    out_df.to_csv('test_predictions.csv', index=False)
    print("Saved test predictions to test_predictions.csv")

# ------------------------
# Entry point
# ------------------------
if __name__ == '__main__':
    main()
