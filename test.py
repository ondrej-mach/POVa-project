from os import system

epochs = 30
print("TEST: Running detector.py")
system(f"python ./detector.py " + str(epochs))

image_path = "data/images/Cars0.png"

print("TEST: Running parser.py")
with open('output/latest.txt', 'r') as f:
    model_path = f.read()
    system(f"python ./predictor.py " + image_path + " " + model_path)
