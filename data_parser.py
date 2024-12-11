import kagglehub
import shutil
import os
import xml.etree.ElementTree as ET

# download the dataset from kaggle (https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
path = kagglehub.dataset_download("andrewmvd/car-plate-detection")

# move to working directory
shutil.copytree(path, "data", dirs_exist_ok=True)

imgPath = "data/images"
annPath = "data/annotations"

# grab the paths to the images
annPaths = list(os.listdir(annPath))
imgPaths = list(os.listdir(imgPath))

# print("Number of images: ", len(imgPaths))
# print("Number of annotations: ", len(annPaths))

# read the xml files and extract the bounding boxes
for annotation in annPaths:
    with open(os.path.join(annPath, annotation)) as file:
        # parse xml annotation
        data = file.read()
        root = ET.fromstring(data)

        # print("FILE: ", annotation)

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

            # print("BOX: [" , coords[0], coords[1], coords[2], coords[3], "]")

            # TODO: further processing required
