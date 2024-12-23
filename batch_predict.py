import os
from PIL import Image

# this is a script that runs the predictor.py script on all images in a folder

model_path = f"output/ok/2024-12-23_11-07-23/"
custom_images_path = 'data/images'
enhance = 0.1

for image in os.listdir(custom_images_path):
    image_path = os.path.join(custom_images_path, image)
    image = Image.open(image_path).convert("RGB")

    print("TEST: Running parser.py")
    os.system(f"python ./predictor.py " + image_path + " " + model_path + " -e " + str(enhance) + " -s 0") 
        
