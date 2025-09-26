import os
import numpy as np
import matplotlib.pyplot as plt
import random

RootDir = "C:/Users/josia/Kanji-Recognition/ETL9B/ETL9B"
RecordsPerFile = 121440

def load():

    RecordSize = 576
    images = []
    labels = []

    # For all 5 data files
    for FileIndex in range(1, 6):

        # Open the file to read
        with open(os.path.join(RootDir, f'ETL9B_{FileIndex}'), "rb") as CurrentFile:

            # Read dummy record
            CurrentFile.read(RecordSize)
            
            # Read all records
            for Record in range(RecordsPerFile):
                
                # Get sheet serial number
                SerialSheetNumber = CurrentFile.read(2)
                
                # Get kanji code (label)
                JISKanjiCode = CurrentFile.read(2)
                JISKanjiCode = int.from_bytes(JISKanjiCode, byteorder="big", signed=False)
                
                if JISKanjiCode > 9331:
                    labels.append(JISKanjiCode)
                else:
                    CurrentFile.read(4) # In place of JIS typical reading
                    CurrentFile.read(504) # In place of image data reading
                    CurrentFile.read(64) # In place of uncertain data
                    continue

                # Get typical reading
                JISTypicalReading = CurrentFile.read(4)

                # Get Images
                ImageData = CurrentFile.read(504)
                TempImageInts = np.frombuffer(ImageData, dtype=np.uint8)
                TempImageBits = np.unpackbits(TempImageInts)
                ImagePixels = TempImageBits * 255
                images.append(ImagePixels.reshape((63, 64)))

                # Get uncertain data
                UncertainData = CurrentFile.read(64)
    

    return np.array(images), labels


images, labels = load()
past = set()
for img, lbl in zip(images, labels):
    
    past.add(lbl)
    if len(past) == 76:
        plt.imshow(img, cmap='gray')
        plt.savefig('magiccharacter.png')
        print('Saved character 75')
