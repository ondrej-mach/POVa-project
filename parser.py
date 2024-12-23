import kagglehub
import shutil
import os
import torch

import utils


def parse_data(device):

    # Download the dataset from kaggle (https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
    path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
    shutil.copytree(path, "data", dirs_exist_ok=True)

    imgPath = "data/images"
    annPath = "data/annotations"
    annPaths = list(os.listdir(annPath))
    dataset = []

    # Read the xml files and extract the bounding boxes
    for annotation in annPaths:
        with open(os.path.join(annPath, annotation)) as file:
            # Parse xml annotation
            data = file.read()
            root, coords = utils.extract_bounding_box(data, True)
            if coords is None:
                continue
            coords_tensor = torch.tensor(coords).float().to(device)

            img_path = os.path.join(imgPath, root.find("filename").text)

            # Preprocess the image
            image_tensor, _ = utils.preprocess_image(img_path, device)

            dataset.append({"img": image_tensor, "box": coords_tensor})

    return dataset
