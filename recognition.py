from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

# this is only a demonstration of how to process the extracted plates using TrOCR
# modified from https://huggingface.co/microsoft/trocr-base-printed
# limitation is that the model is not trained on license plates, so the results may not be accurate
# it cannot read multi line plates, is negatively affected by the plate's relative scale in the cropped image

custom_images_path = 'output/images'

for image in os.listdir(custom_images_path):
    image_path = os.path.join(custom_images_path, image)
    image = Image.open(image_path).convert("RGB")

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Predicted text: {generated_text}")

    # save the text to a file
    with open(image_path.replace(image_path, "output/ocr_pred.txt"), "a") as file:
        result = image_path.split("/")[-1].split("\\")[-1]
        file.write(result + ": " + generated_text + "\n")
