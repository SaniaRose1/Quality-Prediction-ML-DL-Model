import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error




CSV_PATH = "dataset/tomatoes_dataset.csv"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

MAX_SHELF_LIFE_DAYS = 14.0

MODEL_PATH = "models/tomato_quality_model.keras"




df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))




_, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

_, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

test_df = test_df.reset_index(drop=True)

print("Test images:", len(test_df))




test_df["quality_target"] = (
    test_df["quality_score"] / 100.0
)

test_df["shelf_target"] = (
    test_df["self_life_days"]
    / MAX_SHELF_LIFE_DAYS
)

test_df["spoilage_target"] = (
    test_df["spoilage_risk"]
)




def load_image(
    image_path,
    quality,
    shelf,
    spoilage
):

    image = tf.io.read_file(image_path)

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




test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["quality_target"].astype("float32").values,
        test_df["shelf_target"].astype("float32").values,
        test_df["spoilage_target"].astype("float32").values
    )
)

test_dataset = test_dataset.map(
    load_image
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)




print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)




predictions = model.predict(
    test_dataset
)


quality_predictions = (
    predictions[0].flatten()
    * 100
)

shelf_predictions = (
    predictions[1].flatten()
    * MAX_SHELF_LIFE_DAYS
)

spoilage_predictions = (
    predictions[2].flatten()
    * 100
)




actual_quality = (
    test_df["quality_score"].values
)

actual_shelf = (
    test_df["self_life_days"].values
)

actual_spoilage = (
    test_df["spoilage_risk"].values
    * 100
)





quality_mae = mean_absolute_error(
    actual_quality,
    quality_predictions
)

quality_rmse = np.sqrt(
    mean_squared_error(
        actual_quality,
        quality_predictions
    )
)


shelf_mae = mean_absolute_error(
    actual_shelf,
    shelf_predictions
)

shelf_rmse = np.sqrt(
    mean_squared_error(
        actual_shelf,
        shelf_predictions
    )
)


spoilage_mae = mean_absolute_error(
    actual_spoilage,
    spoilage_predictions
)

spoilage_rmse = np.sqrt(
    mean_squared_error(
        actual_spoilage,
        spoilage_predictions
    )
)




print("\n")
print("=" * 50)
print("          MODEL EVALUATION")
print("=" * 50)

print("\nQUALITY")
print(
    f"MAE  : {quality_mae:.2f} points"
)

print(
    f"RMSE : {quality_rmse:.2f} points"
)


print("\nSELF-LIFE")

print(
    f"MAE  : {shelf_mae:.2f} days"
)

print(
    f"RMSE : {shelf_rmse:.2f} days"
)


print("\nSPOILAGE RISK")

print(
    f"MAE  : {spoilage_mae:.2f}%"
)

print(
    f"RMSE : {spoilage_rmse:.2f}%"
)


print("\n")
print("=" * 50)
     