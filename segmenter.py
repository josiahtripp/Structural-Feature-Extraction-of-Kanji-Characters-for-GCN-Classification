import SegmenterTraining as st
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


'''
for outer_i in range(11337):

    # Load or create 10 images for this character
    images = [np.array(stroke_groups[outer_i][x]) for x in range(10)]

    # Create figure with 2x6 subplots (12 total)
    fig, axes = plt.subplots(2, 6, figsize=(20, 4))
    axes = axes.ravel()  # Flatten to 1D array for easy indexing

    # Iterate through each subplot
    for idx, ax in enumerate(axes):
        if idx > 10:  # we only have 11 images total
            ax.axis('off')
            continue

        if idx == 0:
            ax.imshow(np.array(characters[outer_i]), cmap='gray')
            ax.set_title('Whole character')
        else:
            ax.imshow(images[idx - 1], cmap='gray')
            ax.set_title(f'Group {idx - 1}')

        ax.axis('off')

    plt.tight_layout()
    plt.show()
'''