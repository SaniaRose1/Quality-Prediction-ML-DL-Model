import os
import sys
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)




MODEL_PATH = (
    "models/tomato_quality_model.keras"
)

SETTINGS_PATH = (
    "models/model_settings.json"
)




print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)




with open(
    SETTINGS_PATH,
    "r"
) as f:

    settings = json.load(f)


IMAGE_SIZE = tuple(
    settings["image_size"]
)

MAX_SHELF_LIFE_DAYS = (
    settings["max_shelf_life_days"]
)




def predict_tomato(
    image_path
):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


   

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )


    

    image_array = img_to_array(
        image
    )


   

    image_array = (
        image_array / 255.0
    )


    

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


   

    predictions = model.predict(
        image_array,
        verbose=0
    )


   

    quality_normalized = float(
        predictions[0][0][0]
    )

    quality_score = (
        quality_normalized * 100
    )


   

    shelf_normalized = float(
        predictions[1][0][0]
    )

    shelf_life_days = (
        shelf_normalized
        * MAX_SHELF_LIFE_DAYS
    )


   

    spoilage_normalized = float(
        predictions[2][0][0]
    )

    spoilage_percentage = (
        spoilage_normalized * 100
    )


    return (
        quality_score,
        shelf_life_days,
        spoilage_percentage
    )




if len(sys.argv) < 2:

    print(
        "\nUsage:"
    )

    print(
        "python predict.py path_to_image"
    )

    print(
        "\nExample:"
    )

    print(
        'python predict.py "dataset/images/freshTomato/tomato1.jpg"'
    )

    sys.exit()


image_path = sys.argv[1]




quality, shelf_life, spoilage = (
    predict_tomato(
        image_path
    )
)




print(
    "\n"
    + "=" * 45
)

print(
    "        TOMATO QUALITY PREDICTION"
)

print(
    "=" * 45
)

print(
    f"Quality Score       : {quality:.2f} / 100"
)

print(
    f"Estimated Self-Life : {shelf_life:.2f} days"
)

print(
    f"Spoilage Risk       : {spoilage:.2f}%"
)

print(
    "=" * 45
)