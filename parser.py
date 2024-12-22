import kagglehub
import shutil
import os
from torchvision import transforms
import PIL
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
        try:
            with open(os.path.join(annPath, annotation)) as file:
                # Parse xml annotation
                data = file.read()

                root, coords = utils.extract_bounding_box(data, True)

                img_path = os.path.join(imgPath, root.find("filename").text)

                # Load and preprocess the image
                image = PIL.Image.open(img_path).convert("RGB")
                image = transforms.Resize((416, 416))(image)
                image = transforms.ToTensor()(image).to(device)
                image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

                coords_tensor = torch.tensor(coords).float().to(device)
                dataset.append({"img": image, "box": coords_tensor})

        except Exception as e:
            print(f"Error processing annotation: {annotation}", e)
            continue

    return dataset
