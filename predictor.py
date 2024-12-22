import net_model
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import xml.etree.ElementTree as ET
import torchshow as ts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = net_model.CnnModel(0.12).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set model to evaluation mode
    return model

def predict_bounding_box(model, image_tensor):
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(image_tensor).cpu().numpy()
    return prediction[0] 

def visualize_prediction(image, bbox_pred, bbox_true):
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((bbox_true[0], bbox_true[1]), 
                                        bbox_true[2] - bbox_true[0], 
                                        bbox_true[3] - bbox_true[1], 
                                        linewidth=2, edgecolor='g', facecolor='none', label='Truth'))
    plt.gca().add_patch(plt.Rectangle((bbox_pred[0], bbox_pred[1]),
                                        bbox_pred[2] - bbox_pred[0], 
                                        bbox_pred[3] - bbox_pred[1], 
                                        linewidth=2, edgecolor='r', facecolor='none', label='Prediction'))
    plt.axis('off')
    plt.legend()
    plt.show()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # Resize to match training input size
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    print(image.size)
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    return image_tensor, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("img_path", type=str, help="Path to the image")
    args = parser.parse_args()
   
    # Load the trained model
    model = load_model(args.model_path)

    # Specify the image path for prediction
    image_path = args.img_path

    # Preprocess the image and make predictions
    image_tensor, original_image = preprocess_image(image_path)
    predicted_bbox = predict_bounding_box(model, image_tensor)
    print("Predicted bounding box:            ", predicted_bbox)

    # Load the ground truth bounding box
    annotation_file = image_path.replace("images", "annotations").replace(".png", ".xml")
    with open(annotation_file) as file:
        data = file.read()
        root = ET.fromstring(data)
        box = root.find("object").find("bndbox")
        truth_bbox = [
            int(box.find("xmin").text),
            int(box.find("ymin").text),
            int(box.find("xmax").text),
            int(box.find("ymax").text),
        ]
        print("Ground truth bounding box:         ", truth_bbox)

        img_size = root.find("size")
        img_width = int(img_size.find("width").text)
        img_height = int(img_size.find("height").text)

    norm_bbox = (predicted_bbox * [img_width, img_height, img_width, img_height]).astype(int)
    print("Scaled predicted bounding box:     ", norm_bbox)

    visualize_prediction(original_image, norm_bbox, truth_bbox)

    #sqrt_bbox = (predicted_bbox * 416).astype(int)
    #print("Scaled bounding box:                   ", sqrt_bbox)

    # Display the square image with the predicted bounding box 
    '''ts.save(image_tensor, 'output/image_tensor.png')
    image = plt.imread('output/image_tensor.png')
    _, ax = plt.subplots() 
    ax.imshow(image)
    ax.add_patch(plt.Rectangle((sqrt_bbox[0], 
                                sqrt_bbox[1]), 
                                sqrt_bbox[2] - sqrt_bbox[0], 
                                sqrt_bbox[3] - sqrt_bbox[1], 
                                linewidth=2, edgecolor='r', facecolor='none', label='Prediction'))
    plt.axis('off')
    plt.legend()
    plt.show()'''

if __name__ == "__main__":
    main()