import os
import cv2
import numpy as np
from albumentations import Compose, RandomRotate90, HorizontalFlip, Transpose, RandomBrightnessContrast, ShiftScaleRotate, VerticalFlip, ElasticTransform, GridDistortion, GaussNoise, RandomResizedCrop, Affine

def augment_images(input_dir, output_dir, num_augmentations=5):
    """
    Аугментация всех изображений в папке с сохранением черного фона и логированием прогресса.

    Args:
        input_dir (str): Путь к папке с исходными изображениями.
        output_dir (str): Путь к папке для сохранения аугментированных изображений.
        num_augmentations (int): Количество аугментированных изображений на одно исходное.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)

    augmentations = Compose([
        RandomRotate90(p=1.0),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.8),
        Affine(translate_percent=(0.2, 0.2), scale=(0.7, 1.3), rotate=(-45, 45), p=0.9),
        ElasticTransform(alpha=1, sigma=50, p=0.7),
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.7),
        GaussNoise(p=0.5),
        RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), p=0.8)
    ])

    for idx, image_name in enumerate(images):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Не удалось загрузить изображение: {image_name}")
            continue

        mask = (image != 0).astype(np.uint8)

        for i in range(num_augmentations):
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            augmented = augmentations(image=image, mask=mask_resized)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']
            augmented_image[augmented_mask == 0] = 0

            output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_aug_{i}.png")
            cv2.imwrite(output_path, augmented_image)

        progress = ((idx + 1) / total_images) * 100
        print(f"Прогресс: {progress:.2f}% ({idx + 1}/{total_images})")

if __name__ == "__main__":
    input_directory = "/home/user/leaf-analyzer/images/disease_plants"
    output_directory = "/home/user/leaf-analyzer/images/augmented_disease_plants"
    augment_images(input_directory, output_directory, num_augmentations=5)