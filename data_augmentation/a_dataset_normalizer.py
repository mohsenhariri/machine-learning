import os
from PIL import Image

input_path = r"./data/custom_cat_dog"  # in linux "/" isn't matter!

for file in os.listdir(input_path):
    if file.endswith(".jpg"):
        img_path = input_path + "/" + file
        img = Image.open(img_path)
        img = img.resize((300, 300))
        img_out_path = input_path + "/resize/" + file
        img.save(img_out_path)
