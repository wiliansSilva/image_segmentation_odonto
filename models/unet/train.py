import tensorflow as tf
from tensorflow import keras

from unet import build_unet_model
from utils import Parser, DataAugmentation

if __name__ == "__main__":
    #CLASSES = ['restoration', 'root canal treatment', 'dental implant', 'crown', 'tooth']
    CLASSES = ['root canal treatment']
    DATASET_PATH = 'dataset/'
    WEIGHTS_PATH = 'results/weights'
    LOGGER_PATH = 'results/csv_logger'
    IMAGES_PATH = os.path.join(DATASET_PATH, "images")
    MASKS_PATH = os.path.join(DATASET_PATH, "masks")
    IMAGE_SIZE = 256
    TRAIN_SPLIT = 0.8

    LR = 1e-4
    EPOCHS = 40
    BATCH_SIZE = 5

    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    criterion = keras.losses.CategoricalCrossentropy()

    metrics = [
        keras.metrics.CategoricalAccuracy(),
        keras.metrics.Recall(),
        keras.metrics.AUC()
    ]

    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_PATH, f"ckpt_{TRAIN_ID}.h5"), save_weights_only=True),
        keras.callbacks.CSVLogger(os.path.join(LOGGER_PATH, f"{TRAIN_ID}.csv"), append=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    ]

    # DATASET
    ds = tf.data.Dataset.list_files(os.path.join(IMAGES_PATH, '*'))

    preprocessing = keras.Sequential([
        keras.layers.Normalization(),
        keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)
    ])

    ds = ds.shuffle(buffer_size=100, seed=1234).map(Parser(CLASSES, MASKS_PATH), num_parallel_calls = tf.data.AUTOTUNE)
    # Image normalization
    ds = ds.map(lambda x, y: (preprocessing(x), preprocessing(y)), num_parallel_calls = tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (DataAugmentation()(x), DataAugmentation(True)(y)), num_parallel_calls = tf.data.AUTOTUNE).unbatch()
    ds = ds.map(lambda x, y: (x, tf.cast(y, tf.uint8)), num_parallel_calls = tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, 2)), num_parallel_calls = tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (x, tf.squeeze(y, axis=-2)), num_parallel_calls = tf.data.AUTOTUNE)

    print(f"Dataset size: {len(ds)}")

    model = build_unet_model(input_shape=(256,256,1), n_classes=2)