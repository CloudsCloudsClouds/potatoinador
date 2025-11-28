# One thing I say to gemini, it becomes literally unreadable -_-
import os
import random
import shutil
from pathlib import Path

import kagglehub


def find_all_image_files(directory: str) -> list[str]:
    """Recursively finds all image files in a directory."""
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    return image_files


def copy_files(file_list: list[str], destination_dir: str):
    """Copies a list of files to a destination directory."""
    os.makedirs(destination_dir, exist_ok=True)
    for file_path in file_list:
        try:
            shutil.copy(
                file_path, os.path.join(destination_dir, os.path.basename(file_path))
            )
        except Exception as e:
            print(f"Could not copy file {file_path}: {e}")
    print(f"Copied {len(file_list)} files to {destination_dir}")


def prepare_image_datasets(
    potato_dataset_slug: str,
    rock_dataset_slug: str,
    neither_dataset_slug: str,
    output_base_dir: str = "data/combined_dataset",
    num_rock_samples: int = 1000,
    num_neither_samples: int = 1000,
):
    """
    Downloads datasets, organizes them into a three-class (potato, rock, neither)
    dataset directory, and samples images for the rock and neither classes.
    """
    # --- Download Datasets ---
    print("Downloading datasets from KaggleHub (this might take a while)...")
    potato_download_path = kagglehub.dataset_download(potato_dataset_slug)
    rock_download_path = kagglehub.dataset_download(rock_dataset_slug)
    neither_download_path = kagglehub.dataset_download(neither_dataset_slug)
    print("Downloads complete.")

    # --- Define Output Directories ---
    output_potato_dir = os.path.join(output_base_dir, "potato")
    output_rock_dir = os.path.join(output_base_dir, "rock")
    output_neither_dir = os.path.join(output_base_dir, "neither")

    # --- Process Potato Images ---
    print("\nProcessing potato images...")
    # This specific dataset has a nested structure we need to navigate
    potato_images_path = os.path.join(
        potato_download_path, "Vegetable Images", "train", "Potato"
    )
    potato_image_files = find_all_image_files(potato_images_path)
    copy_files(potato_image_files, output_potato_dir)

    # --- Process Rock Images ---
    print("\nProcessing rock images...")
    all_rock_image_files = find_all_image_files(rock_download_path)
    if len(all_rock_image_files) < num_rock_samples:
        print(
            f"Warning: Only {len(all_rock_image_files)} rock images found, requested {num_rock_samples}. Using all available."
        )
        selected_rock_images = all_rock_image_files
    else:
        selected_rock_images = random.sample(all_rock_image_files, num_rock_samples)
    copy_files(selected_rock_images, output_rock_dir)

    # --- Process Neither Images ---
    print("\nProcessing 'neither' images...")
    all_neither_image_files = find_all_image_files(neither_download_path)
    if len(all_neither_image_files) < num_neither_samples:
        print(
            f"Warning: Only {len(all_neither_image_files)} 'neither' images found, requested {num_neither_samples}. Using all available."
        )
        selected_neither_images = all_neither_image_files
    else:
        selected_neither_images = random.sample(
            all_neither_image_files, num_neither_samples
        )
    copy_files(selected_neither_images, output_neither_dir)

    print("\nDataset preparation complete!")
    print(f"Final dataset structure created in '{output_base_dir}'")


if __name__ == "__main__":
    prepare_image_datasets(
        potato_dataset_slug="misrakahmed/vegetable-image-dataset",
        rock_dataset_slug="neelgajare/rocks-dataset",
        neither_dataset_slug="mathurinache/gpr1200-dataset",
        num_rock_samples=1000,
        num_neither_samples=1000,
    )
