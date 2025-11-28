import os
import random
import shutil

import kagglehub


def prepare_image_datasets(
    potato_dataset_slug: str,
    rock_dataset_slug: str,
    output_base_dir: str = "potatoinador/data/combined_dataset",
    num_rock_samples: int = 1000,
):
    """
    Downloads datasets, organizes them into a combined dataset directory,
    and samples a specified number of rock images.

    Args:
        potato_dataset_slug (str): The KaggleHub slug for the potato dataset.
        rock_dataset_slug (str): The KaggleHub slug for the rock dataset.
        output_base_dir (str): The base directory within your project
                                where the organized dataset will be created.
        num_rock_samples (int): The number of rock images to sample.
    """
    print("Downloading datasets from KaggleHub...")
    potato_download_path = kagglehub.dataset_download(potato_dataset_slug)
    rock_download_path = kagglehub.dataset_download(rock_dataset_slug)

    potato_images_path = os.path.join(
        potato_download_path, "Vegetable Images", "train", "Potato"
    )

    print(f"Potato dataset downloaded to: {potato_images_path}")
    print(f"Rock dataset downloaded to: {rock_download_path}")

    output_potato_dir = os.path.join(output_base_dir, "potato")
    output_rock_dir = os.path.join(output_base_dir, "rock")

    # Ensure output directories exist (already created by the model, but good practice)
    os.makedirs(output_potato_dir, exist_ok=True)
    os.makedirs(output_rock_dir, exist_ok=True)

    print(f"\nCopying potato images to {output_potato_dir}...")
    potato_image_files = []
    for root, _, files in os.walk(potato_images_path):
        for file in files:
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
            ):
                potato_image_files.append(os.path.join(root, file))

    for img_path in potato_image_files:
        shutil.copy(
            img_path, os.path.join(output_potato_dir, os.path.basename(img_path))
        )
    print(f"Copied {len(potato_image_files)} potato images.")

    print(
        f"\nSampling and copying {num_rock_samples} rock images to {output_rock_dir}..."
    )
    all_rock_image_files = []
    for root, _, files in os.walk(rock_download_path):
        for file in files:
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
            ):
                all_rock_image_files.append(os.path.join(root, file))

    if len(all_rock_image_files) < num_rock_samples:
        print(
            f"Warning: Only {len(all_rock_image_files)} rock images found, requested {num_rock_samples}. Copying all available."
        )
        selected_rock_images = all_rock_image_files
    else:
        selected_rock_images = random.sample(all_rock_image_files, num_rock_samples)

    for img_path in selected_rock_images:
        shutil.copy(img_path, os.path.join(output_rock_dir, os.path.basename(img_path)))
    print(f"Copied {len(selected_rock_images)} rock images.")
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    prepare_image_datasets(
        potato_dataset_slug="misrakahmed/vegetable-image-dataset",
        rock_dataset_slug="neelgajare/rocks-dataset",
        num_rock_samples=1000,
    )
