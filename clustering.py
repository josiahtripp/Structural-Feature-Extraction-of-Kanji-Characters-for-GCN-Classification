import ETL9B
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import os
import random

def GetComponents(X):

    print(f'Computing covariance matrix of images Numpy-array...')
    X -= X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / (X.shape[0] - 1)
    print(C.shape)

    print(f'Performing eigen decomposition on covariance matrix...')
    eigenvalues, V = np.linalg.eigh(C)  # V are eigenvectors

    idk = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idk]
    V = V[:, idk]
    
    print(f'Saving PCA components to data file...')

    # Compute per-component min/max
    pc_min = V.min(axis=0, keepdims=True)
    pc_max = V.max(axis=0, keepdims=True)

    eps = 1e-8
    pc_img = 255 * (V - pc_min) / (pc_max - pc_min + eps)  # still (n_components, H, W)

    print(f'Square-root eigenvalues to extract singular values...')
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))  # avoid small negatives

    print(f'Sorting and normalizing singular values')
    singular_values = np.sort(singular_values)[::-1]
    singular_values /= np.sum(singular_values)
    
    return pc_img, singular_values

def GetSpheres(T, labels, SVs):

    centers = defaultdict(list)
    centroids = []
    radii = []

    for vector, label in zip(T, labels):
        centers[label].append(vector)

    for label in centers:
        points = np.stack(centers[label], axis=0)
        centroid = np.mean(points, axis=0)
        centroids.append(centroid)

        diff = points - centroid                # (n_samples, n_components)
        weighted = diff * SVs[:diff.shape[1]]   # scale each dimension
        distances = np.linalg.norm(weighted, axis=1)
        radii.append(np.mean(distances))

    
    return np.array(centroids), radii

def GetClusterGraph(centroids, radii, SVs):

    n = len(centroids)
    centroids = np.array(centroids)
    radii = np.array(radii)

    ClusterGraph = nx.Graph()
    ClusterGraph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i+1, n):
            diff = centroids[i] - centroids[j]              # (n_samples, n_components)
            weighted = diff * SVs[:diff.shape[0]]   # scale each dimension
            L2 = np.linalg.norm(weighted)

            if L2 <= radii[i] + radii[j]:
                ClusterGraph.add_edge(i, j)

    return ClusterGraph

def DrawClusterGraph(ClusterGraph, dim):

    # find connected components (clusters)
    clusters = list(nx.connected_components(ClusterGraph))
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))  # distinct colors

    # assign color per node based on its cluster
    color_map = {}
    for c_idx, cluster in enumerate(clusters):
        for node in cluster:
            color_map[node] = colors[c_idx]

    node_colors = [color_map[node] for node in ClusterGraph.nodes()]

    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(ClusterGraph, seed=42)  # force-directed layout

    nx.draw_networkx_nodes(ClusterGraph, pos, node_size=10, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(ClusterGraph, pos, width=0.2, alpha=0.3)

    filename = os.path.join(OutputDir, f"{dim}-dimension")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=60)
    plt.close()
    print(f"Cluster graph saved to {filename}")

images, labels = ETL9B.load()

# Flatten images
print(f'Flattening images Numpy-array from shape {images.shape} to ({images.shape[0]}, {images.shape[1] * images.shape[2]})...')
X = images.reshape(len(images), -1)

# Convert type
print(f'Converting images Numpy-array to float32...')
X = X.astype(np.float32) / 255.0
V, SVs = GetComponents(X)

OutputDir = 'Cluster Graphs'
os.makedirs(OutputDir, exist_ok=True)

ClusterCounts = []

dims = list(range(1, 4033))

random.seed(42)

while len(dims) > 0:


    selection = 2821
    dims.remove(selection)
    i = selection

    print(f"Calculating T{i}...")
    T = X @ V[:, :i]
    SVs = SVs[:i]

    print(f"Calculating centroids and radii")
    centroids, radii = GetSpheres(T, labels, SVs)

    import matplotlib.pyplot as plt
    
    cmap = plt.cm.get_cmap('hsv', centroids.shape[0])

    # Extract x and y coordinates from centroids
    x_coords = centroids[:, 0]
    y_coords = centroids[:, 1]

    # Create a scatter plot
    # The 'c' argument takes an array of values that will be mapped to colors by the colormap
    # np.arange(num_centroids) provides unique values for each centroid, ensuring unique colors
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, c=np.arange(centroids.shape[0]), cmap=cmap, s=10) # 's' controls marker size

    # Add labels and title
    plt.xlabel('Distance from first component')
    plt.ylabel('Distance from second component')
    plt.title('2 Dimension Projection of Images')

    # Add a colorbar to understand the color mapping
    plt.colorbar(label='Centroid Index')


    plt.axis("equal")
    plt.savefig("2-dim-proj.png")


    print("Mean centroid distance:", np.mean([np.linalg.norm(a-b) for a in centroids for b in centroids]))
    print("Mean radius:", np.mean(radii))


    print("Generating cluster graph...")
    ClusterGraph = GetClusterGraph(centroids, radii, SVs)
    CC = nx.number_connected_components(ClusterGraph)
    DrawClusterGraph(ClusterGraph, i)

    c = nx.connected_components(ClusterGraph)
    for i, comp in enumerate(c):
        print(f"Component {i}: {list(comp)}")

    print(f"Found {CC} Clusters")
    ClusterCounts.append(CC)

HighCluster = list.index(max(ClusterCounts))

with open("ClusterInfo.txt", "w") as fp:
    for idx, clusters in enumerate(ClusterCounts):
        fp.write(f"{idx+1} dimension projection yielded {clusters} clusters\n")

    #DrawClusterGraph(ClusterGraph, i)
    #tw = input(f"Cluster Graph Generated for {i}-dim projection\nPress any key to continue...")