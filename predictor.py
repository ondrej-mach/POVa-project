import torch
import argparse
import matplotlib.pyplot as plt
import PIL.Image as Image

import utils, net_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and its weights
def load_model(model_path):
    model = net_model.CnnModel(0).to(device)
    print("Loading model from: ", model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


# Make predictions on the image
def predict_bounding_box(model, image_tensor, img_size, enh):
    model.eval()
    with torch.no_grad():
        # Make prediction, denormalize the bounding box and convert it to an integers
        prediction = model(image_tensor).cpu().numpy()
        formatted_pred = [f"{value:.4f}" for value in prediction[0]]
        print("Predicted bounding box: ", formatted_pred)
        denormalized_bbox = prediction[0] * [
            img_size[0],
            img_size[1],
            img_size[0],
            img_size[1],
        ]
        denorm_bbox = list(map(int, denormalized_bbox))

        # Enhance the bounding box if necessary
        bbox_mid = [(denorm_bbox[0] + denorm_bbox[2]) / 2, (denorm_bbox[1] + denorm_bbox[3]) / 2]
        bbox_enhanced = [
            denorm_bbox[0] - enh * (bbox_mid[0] - denorm_bbox[0]),
            denorm_bbox[1] - enh * (bbox_mid[1] - denorm_bbox[1]),
            denorm_bbox[2] + enh * (denorm_bbox[2] - bbox_mid[0]),
            denorm_bbox[3] + enh * (denorm_bbox[3] - bbox_mid[1]),
        ]
        # Check if the bounding box is within the image, crop if necessary
        if bbox_enhanced[0] < 0:
            bbox_enhanced[0] = 0
        if bbox_enhanced[1] < 0:
            bbox_enhanced[1] = 0
        if bbox_enhanced[2] > img_size[0]:
            bbox_enhanced[2] = img_size[0]
        if bbox_enhanced[3] > img_size[1]:
            bbox_enhanced[3] = img_size[1]

    return bbox_enhanced


# Load the ground truth bounding box and denormalize it, mainly for debugging
def get_orig_bbox(image_path):
    annot_file = image_path.replace("images", "annotations").replace(".png", ".xml")

    with open(annot_file) as file:
        data = file.read()

        _, truth_bbox = utils.extract_bounding_box(data, True)
        truth_bbox = [f"{value:.4f}" for value in truth_bbox]
        print(f"Truth bounding box:      {truth_bbox}")

        _, denorm_truth_bbox = utils.extract_bounding_box(data, False)
        print(f"Truth denorm box:        {denorm_truth_bbox}")

    # return only the denormalized bbox
    return denorm_truth_bbox


# Visualize the predicted and ground truth bounding boxes
def visualize_prediction(image, bbox_pred, bbox_true, show):
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
            edgecolor="m",
            facecolor="none",
            label="Prediction",
        )
    )
    plt.axis("off")
    plt.legend()
    if show:
        plt.show()


# Save the cropped image
def save_predicted_image(image_path, bbox_pred):
    image = Image.open(image_path)
    cropped_image = image.crop(bbox_pred)
    cropped_image.save(image_path.replace(".png", "_pred.png").replace("data/images", "output/images"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to the image")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("--enhance", '-e', type=float, default="0", help="Enhance the bounding box")
    parser.add_argument("--show", "-s", type=int, default="0", help="Show the image")
    args = parser.parse_args()

    # Load the trained model and its weights from file
    model = load_model(args.model_path + "/model.pth")

    # Specify the image for bbox prediction
    image_path = args.img_path

    enh = args.enhance

    # Preprocess the image
    image_tensor, original_image = utils.preprocess_image(image_path, device)
    img_size = original_image.size

    # Make predictions
    denorm_bbox = predict_bounding_box(model, image_tensor, img_size, enh)
    print("Predicted denorm box:   ", denorm_bbox)

    if denorm_bbox is None:
        print("No bounding box found")
        return

    if denorm_bbox[0] > denorm_bbox[2] or denorm_bbox[1] > denorm_bbox[3]:
        print("Invalid bounding box")
        return

    # Get the original bounding box
    truth_bbox = get_orig_bbox(image_path)

    # Visualize the bounding boxes
    visualize_prediction(original_image, denorm_bbox, truth_bbox, args.show)

    # Save the cropped image
    save_predicted_image(image_path, denorm_bbox)

if __name__ == "__main__":
    main()
