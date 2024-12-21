import net_model
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = net_model.CnnModel(0.12).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set model to evaluation mode
    return model

def predict_bounding_box(model, image_tensor):
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(image_tensor).cpu().numpy()  # Get predictions and move to CPU
    return prediction[0] 

def draw_bounding_box(image, bbox):
    # Convert normalized coordinates back to pixel values
    img_width, img_height = image.size
    xmin, ymin, xmax, ymax = bbox * [img_width, img_height, img_width, img_height]
    print(bbox)
    print(img_width, img_height)
    print(xmin, ymin, xmax, ymax)

    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'))
    plt.axis('off')
    plt.show()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # Resize to match training input size
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
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

    # Draw the predicted bounding box on the original image
    draw_bounding_box(original_image, predicted_bbox)

if __name__ == "__main__":
    main()