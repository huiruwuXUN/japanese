import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.morphology import medial_axis
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.filters import gaussian

# Zhang-Suen ---------------------------------------------------------------------------------------------------------
def zhang_suen_thinning(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    def neighbours(x, y, image):
        img = image
        return [img[x - 1, y], img[x - 1, y + 1], img[x, y + 1], img[x + 1, y + 1],
                img[x + 1, y], img[x + 1, y - 1], img[x, y - 1], img[x - 1, y - 1]]

    def transitions(neighbours):
        n = neighbours + neighbours[0:1]  # Circular sequence
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

    def zhang_suen_thin(image):
        Image_Thinned = image.copy()  # Copy the image
        changing1 = changing2 = 1  # Variables to track changes
        while changing1 or changing2:  # Loop until no changes
            # Step 1
            changing1 = []
            for x in range(1, image.shape[0] - 1):
                for y in range(1, image.shape[1] - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x, y] == 1 and  # If current pixel is foreground
                            2 <= sum(neighbours(x, y, Image_Thinned)) <= 6 and  # Condition on neighbours
                            transitions(neighbours(x, y, Image_Thinned)) == 1 and  # Single 0-1 transition
                            P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):  # Additional conditions
                        changing1.append((x, y))
            for x, y in changing1: Image_Thinned[x, y] = 0
            # Step 2
            changing2 = []
            for x in range(1, image.shape[0] - 1):
                for y in range(1, image.shape[1] - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x, y] == 1 and
                            2 <= sum(neighbours(x, y, Image_Thinned)) <= 6 and
                            transitions(neighbours(x, y, Image_Thinned)) == 1 and
                            P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                        changing2.append((x, y))
            for x, y in changing2: Image_Thinned[x, y] = 0
        return Image_Thinned

    thinned_image = zhang_suen_thin(binary // 255)
    thinned_image = (thinned_image * 255).astype(np.uint8)

    return thinned_image


# Processing images in folders
def process_images_Zhang(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                output_dir = os.path.dirname(output_path)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Failed to read {input_path}")
                    continue

                skeleton_image = zhang_suen_thinning(image)

                cv2.imwrite(output_path, skeleton_image)
                print(f"Processed {input_path} -> {output_path}")



# morphological_skeleton---------------------------------------------------------------------------------------------------------
def morphological_skeleton(image):
    # Convert to binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Corrosion operations
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    # expansion operation
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Detailed operation
    skeleton = skeletonize(dilated // 255)  # skeletonize expects binary image [0, 1]

    # Convert result to uint8 format
    skeleton_image = (skeleton * 255).astype(np.uint8)

    return skeleton_image


def process_images_morphology(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Read image and convert to grey scale
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Failed to read {input_path}")
                    continue

                # skeleton of an extract
                skeleton_image = morphological_skeleton(image)

                # Save results
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, skeleton_image)
                print(f"Processed {input_path} -> {output_path}")

# contour_skeleton---------------------------------------------------------------------------------------------------------
def contour_skeleton(image):

    # Convert to binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Extracting the Outline
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image for drawing the skeleton
    skeleton_image = np.zeros_like(binary)

    # outline
    for contour in contours:
        for point in contour:
            cv2.circle(skeleton_image, tuple(point[0]), 1, 255, -1)

    return skeleton_image


def process_images_contour(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Read image and convert to grey scale
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Failed to read {input_path}")
                    continue

                # skeleton of an extract
                skeleton_image = contour_skeleton(image)

                # Save results
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, skeleton_image)

# medial_axis_transform ---------------------------------------------------------------------------------------------------------
def medial_axis_transform(image):
    # Convert to binary image
    _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # Calculate the medial axis transformation
    skeleton, distance = medial_axis(binary, return_distance=True)

    # Convert result to uint8 format
    skeleton_image = (skeleton * 255).astype(np.uint8)

    return skeleton_image


def process_images_medial_axis(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, os.path.dirname(relative_path))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Read image and convert to grey scale
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Failed to read {input_path}")
                    continue

                # skeleton of an extract
                skeleton_image = medial_axis_transform(image)

                # Save results
                output_path = os.path.join(output_dir, os.path.basename(file))
                cv2.imwrite(output_path, skeleton_image)

# fast_marching_method ---------------------------------------------------------------------------------------------------------
def fast_marching_method(image):
    # Calculating the gradient
    gimage = inverse_gaussian_gradient(image)

    # Initialising contours: using rectangles
    init_ls = np.zeros(image.shape, dtype=np.uint8)
    init_ls[image.shape[0] // 4:image.shape[0] * 3 // 4, image.shape[1] // 4:image.shape[1] * 3 // 4] = 1

    # Applying morphometric geodesic active contours
    skeleton = morphological_geodesic_active_contour(gimage, num_iter=100, init_level_set=init_ls, smoothing=1,
                                                     balloon=-1)

    # Convert result to uint8 format
    skeleton_image = (skeleton * 255).astype(np.uint8)

    return skeleton_image


def process_images_fast_marching(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Read image and convert to grey scale
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                # skeleton of an extract
                try:
                    skeleton_image = fast_marching_method(image)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                    continue

                # Save results
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, skeleton_image)



# main---------------------------------------------------------------------------------------------------------
def main():

    input_folder = './dataOrigin'
    output_folder = './dataOriginZhang' 

    process_images_Zhang(input_folder, output_folder)
    # process_images_morphology(input_folder, output_folder)
    # process_images_contour(input_folder, output_folder)

    # not work(BELOW)
    # process_images_medial_axis(input_folder, output_folder)
    # process_images_fast_marching(input_folder, output_folder)

if __name__ == '__main__':
    main()