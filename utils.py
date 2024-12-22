from datetime import datetime
import matplotlib.pyplot as plt
import torch
import os

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

    # Get the current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

    # Define the folder name
    folder_name = f"output/{timestamp}"

    # Create the directory
    os.makedirs(folder_name, exist_ok=True)  # exist_ok=True prevents an error if the folder already exists

    print(f"Folder created: {folder_name}")
    return folder_name

def save_model(model, trainHist, folder_name):
    print("Saving model...")
    
    os.makedirs(folder_name, exist_ok=True)

    plot_loss(folder_name, trainHist)
    torch.save(model.state_dict(), folder_name + "/model.pth")
    torch.save(trainHist, folder_name + "/trainHist.pth")
    print("Model saved successfully to " + folder_name + "/model.pth")