import shutil
import os
import glob
import random
import sys
import json

def split_yolov8_dataset(dataset_path, output_path, train_size, val_size):
    # Create the output directories
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)

    # Get the list of files
    image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(dataset_path, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(dataset_path, "*.png")))

    # Shuffle the files
    random.shuffle(image_files)

    # Split the files
    train_files = image_files[:int(train_size * len(image_files))]
    val_files = image_files[int(train_size * len(image_files)):int((train_size + val_size) * len(image_files))]
    test_files = image_files[int((train_size + val_size) * len(image_files)):]

    # Move the files
    for image_file in train_files:
        identifier = os.path.basename(image_file).split(".")[0]
        all_files = glob.glob(os.path.join(dataset_path, identifier + "*"))
        for file in all_files:
            shutil.copy(file, os.path.join(output_path, "train", os.path.basename(file)))

    for image_file in val_files:
        identifier = os.path.basename(image_file).split(".")[0]
        all_files = glob.glob(os.path.join(dataset_path, identifier + "*"))
        for file in all_files:
            shutil.copy(file, os.path.join(output_path, "val", os.path.basename(file)))

    for image_file in test_files:
        identifier = os.path.basename(image_file).split(".")[0]
        all_files = glob.glob(os.path.join(dataset_path, identifier + "*"))
        for file in all_files:
            shutil.copy(file, os.path.join(output_path, "test", os.path.basename(file)))

def split_coco_dataset(dataset_path, output_path, train_size, val_size):
    # Create the output directories
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)

    # Read json file
    coco_dataset = None
    with open(dataset_path, "r") as f:
        coco_dataset = json.load(f)

    # Get annotations by images
    annotations_by_images = {}
    for annotation in coco_dataset["annotations"]:
        if annotation["image_id"] not in annotations_by_images:
            annotations_by_images[annotation["image_id"]] = []
        annotations_by_images[annotation["image_id"]].append(annotation)

    # Split annotations_by_images
    image_ids = list(annotations_by_images.keys())
    
    random.shuffle(image_ids)

    train_image_ids = image_ids[:int(train_size * len(image_ids))]
    val_image_ids = image_ids[int(train_size * len(image_ids)):int((train_size + val_size) * len(image_ids))]
    test_image_ids = image_ids[int((train_size + val_size) * len(image_ids)):]

    # Create train, val, and test annotations
    train_dataset = coco_dataset.copy()
    train_dataset['annotations'] = []
    for train_image_id in train_image_ids:
        train_dataset["annotations"].extend(annotations_by_images[train_image_id])

    val_dataset = coco_dataset.copy()
    val_dataset['annotations'] = []
    for val_image_id in val_image_ids:
        val_dataset["annotations"].extend(annotations_by_images[val_image_id])

    test_dataset = coco_dataset.copy()
    test_dataset['annotations'] = []
    for test_image_id in test_image_ids:
        test_dataset["annotations"].extend(annotations_by_images[test_image_id])

    # Save the datasets
    with open(os.path.join(output_path, "train.json"), "w") as f:
        json.dump(train_dataset, f)

    with open(os.path.join(output_path, "val.json"), "w") as f:
        json.dump(val_dataset, f)
    
    with open(os.path.join(output_path, "test.json"), "w") as f:
        json.dump(test_dataset, f)

def print_help():
    """
    Print help message for this script
    """

    print("Usage: python split_dataset.py <dataset_format> <dataset_path> <output_path> <train_size> <val_size>")
    print("dataset_format: The format of the dataset. It can be either 'yolov8' or 'coco'")
    print("dataset_path: The path to the dataset")
    print("output_path: The path to the output directory")
    print("train_size: The size of the training set. It should be a float between 0 and 1")
    print("val_size: The size of the validation set. It should be a float between 0 and 1")

if __name__ == "__main__":
    dataset_format = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    train_size = float(sys.argv[4])
    val_size = float(sys.argv[5])
    
    if dataset_format == "coco":
        split_coco_dataset(dataset_path, output_path, train_size, val_size)
    elif dataset_format == "yolov8":
        split_yolov8_dataset(dataset_path, output_path, train_size, val_size)