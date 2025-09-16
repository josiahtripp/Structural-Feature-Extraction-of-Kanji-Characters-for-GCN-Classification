import KuzushijiKanji as kk_dataset
import ETL9B as ETL_dataset
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse.linalg import svds
import numpy as np
import os

class Dataset:
    def __init__(self, name: str, LoadingFunction, ImageSize: tuple):
        self.name = name
        self.LoadingFunction = LoadingFunction
        self.ImageSize = ImageSize

def __GetMeanImages(images, labels=None):

    # Overall mean
    if labels is None:

        StackedImages = np.stack(images, axis=0)
        return np.mean(StackedImages, axis=0)
    
    # Label-wise mean
    else:

        # Output lists
        AverageImages = []
        AverageLabels = []

        grouped = defaultdict(list)

        # Group images by label
        for image, label in zip(images, labels):
            grouped[label].append(image)

        # Average by groupings
        for label in grouped:
            grouped[label] = np.stack(grouped[label], axis=0)
            AverageImages.append(np.mean(grouped[label], axis=0))
            AverageLabels.append(label)

        return np.array(AverageImages), AverageLabels

def __GetPCImage(V, idx, ImageSize):

    pc = V[:, idx].reshape(ImageSize[0], ImageSize[1])
    pc_min, pc_max = pc.min(), pc.max()
    pc_img = 255 * (pc - pc_min) / (pc_max - pc_min)
    pc_img = pc_img.astype(np.uint8)

    return pc_img

def __SaveImage(image, FigurePath):

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig(FigurePath, bbox_inches="tight", pad_inches=0)
    plt.close()

def LoadDataset(SD):

    print(f'Loading Images and Labels from {SD.name} dataset...')
    images, labels = SD.LoadingFunction()

    print('\tSuccessfully loaded images and labels from dataset:')
    print(f'\tImage Numpy-array shape: {images.shape}')
    print(f'\tLabel list length: {len(labels)}\n')

    return images, labels

def AverageImages(SD, images, labels):

    # Create output folder if it does not exist
    OutputDir = f'{SD.name}_Mean-Images'
    os.makedirs(OutputDir, exist_ok=True)

    print('Calculating overall mean image...')
    OverallMeanImage = __GetMeanImages(images)

    print(f'\tSaving image to: \"{OutputDir}\"...')
    __SaveImage(OverallMeanImage, os.path.join(OutputDir, f"mean-overall.png"))

    print('Calculating label-wise mean images...')
    MeanImages, MeanLabels = __GetMeanImages(images, labels)
    print(f'\tSuccessfully calculated {len(MeanLabels)} mean images:')
    print(f'\tMean images Numpy-array shape: {MeanImages.shape}')
    print(f'\tMean images label list length: {len(MeanLabels)}')

    print(f'\tSaving images to: \"{OutputDir}\"...\n')

    # Save label-wise mean images
    for img, lbl in zip(MeanImages, MeanLabels):
        __SaveImage(img, os.path.join(OutputDir, f"mean-{lbl}"))

def ComputeComponents(SD, images, OType='data'):

    # Check output type selection
    if OType != 'data' and OType != 'image':
        print(f'Invalid Otype \'{OType}\'')
        return

    # Flatten images
    print(f'Flattening images Numpy-array from shape {images.shape} to ({images.shape[0]}, {images.shape[1] * images.shape[2]})...')
    FlattenedImages = images.reshape(len(images), -1)

    # Convert type
    print(f'Converting images Numpy-array to float32...')
    X = FlattenedImages.astype(np.float32) / 255.0

    print(f'Computing covariance matrix of images Numpy-array...')
    X -= X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / (X.shape[0] - 1)
    print(C.shape)

    print(f'Performing eigen decomposition on covariance matrix...')
    eigenvalues, V = np.linalg.eigh(C)  # V are eigenvectors

    idk = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idk]
    V = V[:, idk]

    if OType == 'image':
        ComponentImagePath = f'{SD.name}_PCA-Comp-Images'
        os.makedirs(ComponentImagePath, exist_ok=True)

        print(f'Saving PCA components as images...')
        for component in range(100):
            img = __GetPCImage(V, component, SD.ImageSize)
            __SaveImage(img, os.path.join(ComponentImagePath, f'Component-{component+1}.png'))

    elif OType == 'data':
        print(f'Saving PCA components to data file...')
        PCAComponents = np.array([__GetPCImage(V, component, SD.ImageSize) for component in range(V.shape[0])])
        np.save(f'{SD.name}-Components.npy', PCAComponents)

    print(f'Square-root eigenvalues to extract singular values...')
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))  # avoid small negatives

    print(f'Sorting and normalizing singular values')
    singular_values = np.sort(singular_values)[::-1]
    singular_values /= np.sum(singular_values)

    print(f'Computing cumulative sum of singular values...')
    cumsum_S = np.cumsum(singular_values)

    CumSumDataPath = f'{SD.name}_SV-CumSum-Data.txt'

    print(f'\tSaving data to: \"{CumSumDataPath}\"')
    with open(CumSumDataPath, "w") as f:
        for SV, per in enumerate(cumsum_S):
            f.write(f"{SV+1}: {(100*per):.5f}%\n")

    print(f'Plotting cumulative sum as graph...')
    plt.figure()
    plt.plot(range(1, len(singular_values)+1), cumsum_S)
    plt.xlabel("Number of Singular Values")
    plt.ylabel("Cumulative Sum")
    plt.title("Cumulative Sum of Normalized Singular Values")
    plt.grid(True)

    FigureName = f'{SD.name}_SV-CumSum-PCA.png'

    print(f'\tSaving graph as image: {FigureName}...')
    plt.savefig(FigureName)

    print(f'Calculating required singular values for data percentages...')
    ReqSVs = []
    for data_percentage in range(100):
        for sv_index, percentage in enumerate(cumsum_S):
            if data_percentage * 0.01 < percentage:
                ReqSVs.append(sv_index)
                break

    FilePath = f'{SD.name}_Required-SVs.txt'

    print(f'\tSaving values to: {FilePath}')
    with open(FilePath, "w") as f:
        for per, SVs in enumerate(ReqSVs):
            f.write(f"{per + 1}% : {SVs+1} Singular Values\n")

# Define available datasets
KuzushijiKanjiDataset = Dataset(name="KuzushijiKanji", LoadingFunction=kk_dataset.load, ImageSize=(64, 64))
ETL9BDataset = Dataset(name="ETL9BDataset", LoadingFunction=ETL_dataset.load, ImageSize=(63, 64))

SD = KuzushijiKanjiDataset
images, labels = LoadDataset(SD)

AverageImages(SD, images, labels)

ComputeComponents(SD, images, OType='data')