import os
import pandas as pd

IMAGE_FOLDER ="dataset/images"
OUTPUT_CSV = "dataset/tomatoes_dataset.csv"

data = []

for class_name in os.listdir(IMAGE_FOLDER):
    class_folder = os.path.join(
        IMAGE_FOLDER,
        class_name
    )

    if not os.path.isdir(class_folder):
        continue

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(
            (".jpg")
        ):

            image_path = os.path.join(
                IMAGE_FOLDER,
                class_name,
                filename
            )

            data.append({
                "image_path": image_path,
                "label": class_name
            })


df = pd.DataFrame(data)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("CSV created successfully!")
print("Total images:", len(df))
print(df.head())
