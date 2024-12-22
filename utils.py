from datetime import datetime
import matplotlib.pyplot as plt
import torch
import os
import xml.etree.ElementTree as ET

def calculate_split(d):
    ratios = [0.8, 0.1, 0.1]
    trCount = int(len(d) * ratios[0])
    vCount = int(len(d) * ratios[1])
    tCount = len(d) - trCount - vCount
    (trData, vData, tData) = torch.utils.data.random_split(
        d,
        [trCount, vCount, tCount],
        generator=torch.Generator().manual_seed(42),
    )
    return trData, vData, tData


def extract_bounding_box(data, normalize):
    root = ET.fromstring(data)
    objects = root.findall("object")
    for obj in objects:
        # Extract the bounding box
        box = obj.find("bndbox")

        # Extract the coordinates
        coords = [
            int(box.find("xmin").text),
            int(box.find("ymin").text),
            int(box.find("xmax").text),
            int(box.find("ymax").text),
        ]

        img_size = root.find("size")
        img_width = int(img_size.find("width").text)
        img_height = int(img_size.find("height").text)

        if normalize:
            # Normalize the coordinates
            coords[0] = coords[0] / img_width
            coords[1] = coords[1] / img_height
            coords[2] = coords[2] / img_width
            coords[3] = coords[3] / img_height

        return root, coords


def plot_loss(folder_name, trainHist):
    os.makedirs("output", exist_ok=True)
    epochs = range(1, len(trainHist["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, trainHist["train_loss"], label="Training Loss")
    plt.plot(epochs, trainHist["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(folder_name + "/loss.png")


def create_timestamped_folder():

    # Get the current date and time, create folder name
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"output/{timestamp}"

    # Create the directory
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder created: {folder_name}")

    return folder_name


def save_model(model, trainHist, folder_name):
    # Create the directory if it does not exist
    os.makedirs(folder_name, exist_ok=True)

    # save model and plotted loss
    torch.save(model.state_dict(), folder_name + "/model.pth")
    torch.save(trainHist, folder_name + "/trainHist.pth")
    plot_loss(folder_name, trainHist)
    print("Model and loss graph saved to " + folder_name)
