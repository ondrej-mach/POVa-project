import net_model
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model and its weights
def load_model(model_path):
    model = net_model.CnnModel(0.12).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


# Make predictions on the image
def predict_bounding_box(model, image_tensor):
    with torch.no_grad():
        # Make prediction, denormalize the bounding box and convert it to an integers
        prediction = model(image_tensor).cpu().numpy()
        denormalized_bbox = prediction[0] * 416
        denormalized_bbox = list(map(int, denormalized_bbox))
    return denormalized_bbox


# Visualize the predicted and ground truth bounding boxes
def visualize_prediction(image, bbox_pred, bbox_true):
    plt.imshow(image)
    plt.gca().add_patch(
        plt.Rectangle(
            (bbox_true[0], bbox_true[1]),
            bbox_true[2] - bbox_true[0],
            bbox_true[3] - bbox_true[1],
            linewidth=2,
            edgecolor="g",
            facecolor="none",
            label="Truth",
        )
    )
    plt.gca().add_patch(
        plt.Rectangle(
            (bbox_pred[0], bbox_pred[1]),
            bbox_pred[2] - bbox_pred[0],
            bbox_pred[3] - bbox_pred[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            label="Prediction",
        )
    )
    plt.axis("off")
    plt.legend()
    plt.show()


# Resize the image and convert it to a tensor
def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    return image_tensor, image


# Load the ground truth bounding box and denormalize it, mainly for debugging
def get_orig_bbox(image_path):
    annotation_file = image_path.replace("images", "annotations").replace(
        ".png", ".xml"
    )

    with open(annotation_file) as file:
        data = file.read()

        _, truth_bbox = utils.extract_bounding_box(data, False)
        print("Truth bounding box:     ", truth_bbox)

    return truth_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("img_path", type=str, help="Path to the image")
    args = parser.parse_args()

    # Load the trained model and its weights from file
    model = load_model(args.model_path + "/model.pth")

    # Specify the image for bbox prediction
    image_path = args.img_path

    # Preprocess the image and make predictions
    image_tensor, original_image = preprocess_image(image_path)
    denorm_bbox = predict_bounding_box(model, image_tensor)
    print("Predicted bounding box: ", denorm_bbox)

    # Get the original bounding box
    truth_bbox = get_orig_bbox(image_path)

    # Visualize the bounding boxes
    visualize_prediction(original_image, denorm_bbox, truth_bbox)


if __name__ == "__main__":
    main()
