import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

def load_images_from_folders(base_folder):
    """
    Load all images from specified base folder.

    Parameters:
    base_folder (str): Path to the base folder containing subfolders.

    Returns:
    tuple: (images, image_paths)
    """
    images = []
    image_paths = []
    for label in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                image_paths.append(img_path)
    return images, image_paths


def extract_features(image):
    """
    Extract SIFT features from an image.

    Parameters:
    image (ndarray): The input image from which to extract features.
                     Expected to be a color image in BGR format.

    Returns:
    ndarray: Array of SIFT descriptors. Each descriptor is a vector
             that represents distinctive features of the image.
    """
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def create_bovw(descriptors, kmeans, num_clusters):
    """
    Create a Bag of Visual Words (BoVW) histogram from SIFT descriptors.

    Parameters:
    descriptors (ndarray): Array of SIFT descriptors extracted from an image.
                           Each descriptor is a vector representing distinctive
                           features of the image.
    kmeans (KMeans): Pre-trained KMeans model used to cluster the descriptors.
                     This model should be fitted on a training set of descriptors.
    num_clusters (int): Number of clusters (visual words) used in the KMeans model.
                        This defines the length of the BoVW histogram.

    Returns:
    ndarray: A histogram representing the frequency of visual words in the image.
             Each bin in the histogram corresponds to a cluster, indicating how
             many descriptors were assigned to that cluster.
    """
    histogram = np.zeros(num_clusters)
    if descriptors is not None:
        clusters = kmeans.predict(descriptors)
        for cluster in clusters:
            histogram[cluster] += 1
    return histogram


def plot_images_by_cluster(image_paths, labels, cluster_id):
    """
    Plot a set of images that belong to a specific cluster.

    Parameters:
    image_paths (list of str): List of file paths to the images.
                               Each path corresponds to an image file.
    labels (list of int): List of cluster labels for each image.
                          Each label corresponds to the cluster assignment of the image.
    cluster_id (int): The cluster ID to filter images by.
                      Only images belonging to this cluster will be displayed.

    Returns:
    None: This function does not return any value. It displays a plot
          with up to 10 images that belong to the specified cluster.
    """
    # Filter images by the specified cluster ID
    cluster_images = [img_path for img_path, label in zip(image_paths, labels) if label == cluster_id]
    num_images = min(len(cluster_images), 10)  # Limit to 10 images
    plt.figure(figsize=(15, 5))  # Create a figure for plotting

    for i, img_path in enumerate(random.sample(cluster_images, num_images)):
        img = cv2.imread(img_path)  # Read the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format
        plt.subplot(1, num_images, i+1)  # Create a subplot for each image
        plt.imshow(img)  # Display the image
        plt.axis('off')  # Hide the axis

    plt.show()  # Show the plot


def image_cluster_sift(imgs_dir, num_clusters=3, cluster_method='kmeans'):
    """
    Clusters images using SIFT features and specified clustering method.

    Parameters:
    - imgs_dir (str): Directory containing images to cluster.
    - num_clusters (int): Number of clusters. Default is 3.
    - cluster_method (str): Clustering method to use. Default is 'kmeans'.

    Returns:
    - tuple: (image_paths, image_labels, image_kmeans)
    """
    # Need to impliment more clustering methods.
    try:
        # Load images
        images, image_paths = load_images_from_folders(imgs_dir)
        print(f"Loaded {len(images)} images")

        # Extract features
        descriptors_list = [extract_features(image) for image in images if extract_features(image) is not None]
        print(f"Extracted descriptors for {len(descriptors_list)} images")

        if not descriptors_list:
            raise ValueError("No descriptors extracted from images.")

        # Stack all descriptors
        all_descriptors = np.vstack(descriptors_list)
        print(f"Stacked all descriptors into array of shape {all_descriptors.shape}")

        # Cluster descriptors
        if cluster_method == 'kmeans':
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)
            print(f"Clustered descriptors into {num_clusters} clusters using KMeans")
        else:
            raise ValueError("Unsupported clustering method. Currently only 'kmeans' is supported.")

        # Create BoVW histograms
        bovw_list = [create_bovw(descriptors, kmeans) for descriptors in descriptors_list]
        print(f"Created BoVW histograms for {len(bovw_list)} images")

        # Convert BoVW histograms to feature matrix
        X = np.array(bovw_list)
        print(f"Feature matrix shape: {X.shape}")

        # Cluster images
        image_kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        image_labels = image_kmeans.labels_

        # Output each image's cluster label
        for path, label in zip(image_paths, image_labels):
            print(f"Image: {path} - Cluster: {label}")

        return image_paths, image_labels, image_kmeans
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

