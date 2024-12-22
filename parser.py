import kagglehub
import shutil
import os
import xml.etree.ElementTree as ET
from torchvision import transforms
import PIL 
import torch
import torchshow as ts


# Define the data preprocessing
def parse_data(device):
    # download the dataset from kaggle (https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
    path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
    shutil.copytree(path, "data", dirs_exist_ok=True)

    imgPath = "data/images"
    annPath = "data/annotations"
    annPaths = list(os.listdir(annPath))
    dataset = []

    # read the xml files and extract the bounding boxes
    for annotation in annPaths:
        try:
            with open(os.path.join(annPath, annotation)) as file:
                # parse xml annotation
                data = file.read()
                root = ET.fromstring(data)

                objects = root.findall("object")
                for obj in objects:

                    # extract the bounding box
                    box = obj.find("bndbox")

                    # extract the coordinates
                    coords = [
                        int(box.find("xmin").text),
                        int(box.find("ymin").text),
                        int(box.find("xmax").text),
                        int(box.find("ymax").text),
                    ]

                    img_size = root.find("size")
                    img_width = int(img_size.find("width").text)
                    img_height = int(img_size.find("height").text)

                    # Normalize the coordinates
                    coords[0] = coords[0] / img_width
                    coords[1] = coords[1] / img_height
                    coords[2] = coords[2] / img_width
                    coords[3] = coords[3] / img_height

                    img_path = os.path.join(imgPath, root.find("filename").text)

                    # Load and preprocess the image
                    image = PIL.Image.open(img_path).convert("RGB")
                    image = transforms.Resize((416, 416))(image)

                    '''if len(dataset) % 100 == 0:
                        draw = PIL.ImageDraw.Draw(image)
                        draw.rectangle(
                            [
                                coords[0] * 416,
                                coords[1] * 416,
                                coords[2] * 416,
                                coords[3] * 416,
                            ],
                            outline="red",
                            width=2,
                        )
                        image.show()'''

                    image = transforms.ToTensor()(image).to(device)
                    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

                    coords_tensor = torch.tensor(coords).float().to(device)
                    dataset.append({"img": image, "box": coords_tensor})

        except Exception as e:
            print(f"Error processing annotation: {annotation}", e)
            continue

    return dataset