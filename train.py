import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model




CSV_PATH = "dataset/tomatoes_dataset.csv"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tomato_quality_model.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


MAX_SHELF_LIFE_DAYS = 14.0




print("\nLoading CSV...")

df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())




required_columns = [
    "image_path",
    "label",
    "quality_score",
    "self_life_days",
    "spoilage_risk"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )




print("\nChecking images...")

df["exists"] = df["image_path"].apply(
    os.path.exists
)

missing = df[~df["exists"]]

if len(missing) > 0:

    print("\nThese image paths do not exist:")

    print(
        missing["image_path"].to_string(
            index=False
        )
    )

    raise FileNotFoundError(
        "Some image files could not be found."
    )


df = df.drop(columns=["exists"])




df["quality_score"] = pd.to_numeric(
    df["quality_score"],
    errors="coerce"
)

df["self_life_days"] = pd.to_numeric(
    df["self_life_days"],
    errors="coerce"
)

df["spoilage_risk"] = pd.to_numeric(
    df["spoilage_risk"],
    errors="coerce"
)




df = df.dropna(
    subset=[
        "quality_score",
        "self_life_days",
        "spoilage_risk"
    ]
).reset_index(drop=True)




if not df["quality_score"].between(
    0, 100
).all():

    raise ValueError(
        "quality_score must be between 0 and 100."
    )


if not (df["self_life_days"] >= 0).all():

    raise ValueError(
        "self_life_days cannot be negative."
    )


if not df["spoilage_risk"].between(
    0, 1
).all():

    raise ValueError(
        "spoilage_risk must be between 0 and 1."
    )


print("\nTotal images:", len(df))




print("\nClass distribution:")

print(
    df["label"].value_counts()
)

print("\nTarget statistics:")

print(
    df[
        [
            "quality_score",
            "self_life_days",
            "spoilage_risk"
        ]
    ].describe()
)





train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)


print("\nDataset split:")

print(
    "Training:",
    len(train_df)
)

print(
    "Validation:",
    len(validation_df)
)

print(
    "Testing:",
    len(test_df)
)




train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

for data in [
    train_df,
    validation_df,
    test_df
]:

    data["quality_target"] = (
        data["quality_score"] / 100.0
    )

    data["shelf_target"] = (
        data["self_life_days"]
        / MAX_SHELF_LIFE_DAYS
    )

    data["spoilage_target"] = (
        data["spoilage_risk"]
    )




def load_image(
    image_path,
    quality,
    shelf,
    spoilage
):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    targets = {
        "quality": quality,
        "shelf_life": shelf,
        "spoilage_risk": spoilage
    }

    return image, targets




def create_dataset(
    dataframe,
    shuffle=False
):

    paths = dataframe[
        "image_path"
    ].values

    quality = dataframe[
        "quality_target"
    ].astype("float32").values

    shelf = dataframe[
        "shelf_target"
    ].astype("float32").values

    spoilage = dataframe[
        "spoilage_target"
    ].astype("float32").values

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            quality,
            shelf,
            spoilage
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_dataset = create_dataset(
    train_df,
    shuffle=True
)

validation_dataset = create_dataset(
    validation_df
)

test_dataset = create_dataset(
    test_df
)




data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.10
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )
])




inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="tomato_image"
)


x = data_augmentation(
    inputs
)


x = layers.Conv2D(
    32,
    (3, 3),
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)


x = layers.Conv2D(
    64,
    (3, 3),
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)


x = layers.Conv2D(
    128,
    (3, 3),
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)


x = layers.Conv2D(
    256,
    (3, 3),
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D()(x)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.4
)(x)



quality_output = layers.Dense(
    1,
    activation="sigmoid",
    name="quality"
)(x)



shelf_life_output = layers.Dense(
    1,
    activation="sigmoid",
    name="shelf_life"
)(x)



spoilage_output = layers.Dense(
    1,
    activation="sigmoid",
    name="spoilage_risk"
)(x)




model = Model(
    inputs=inputs,
    outputs=[
        quality_output,
        shelf_life_output,
        spoilage_output
    ]
)




model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss={
        "quality": "mse",
        "shelf_life": "mse",
        "spoilage_risk": "mse"
    },

    metrics={
        "quality": ["mae"],
        "shelf_life": ["mae"],
        "spoilage_risk": ["mae"]
    }
)




model.summary()




checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)




print("\nStarting training...\n")


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)




print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)




print("\nEvaluating model...")

results = best_model.evaluate(
    test_dataset,
    return_dict=True
)

print("\nTest results:")

for key, value in results.items():

    print(
        f"{key}: {value:.4f}"
    )




settings = {

    "image_size": list(
        IMAGE_SIZE
    ),

    "max_shelf_life_days":
        MAX_SHELF_LIFE_DAYS
}


with open(
    "models/model_settings.json",
    "w"
) as f:

    json.dump(
        settings,
        f,
        indent=4
    )




plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.show()




print(
    "\nTraining completed successfully!"
)

print(
    "Model saved at:",
    MODEL_PATH
)