import os
import numpy as np
from PIL import Image

RootDir = "C:/Users/josia/Kanji-Recognition/kmnist-kanji-dataset/kkanji2"

# Returns N images (NP array shape (N, 64, 64)) and labels (list len() = N)
def load():

    # Output lists
    images = []
    labels = []

    # Walk through all subdirectories
    for label in os.listdir(RootDir):
        SubDir = os.path.join(RootDir, label)
        if not os.path.isdir(SubDir):
            continue # Skip files in root directory'
        
        # Get all image files
        for file in os.listdir(SubDir):
            if file.lower().endswith(".png"):
                FilePath = os.path.join(SubDir, file)


                # Load image in grayscale (64x64)
                image = Image.open(FilePath).convert("L") # "L" = 8-bit grayscale
                ImageArray = np.array(image, dtype=np.uint8) # shape (64, 64)

                images.append(ImageArray)
                labels.append(label)
    
    return np.array(images), labels