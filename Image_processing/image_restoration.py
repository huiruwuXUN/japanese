import cv2
from PIL import Image, ImageEnhance
import numpy as np

# Load the image
image_path = 'character_image.png'  # Replace with your image path
image = cv2.imread(image_path)

# Convert to grayscale (if it's not already)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and improve clarity
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Use adaptive thresholding to enhance the character edges
enhanced_image = cv2.adaptiveThreshold(blurred_image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

# Convert back to PIL image for further enhancement
pil_image = Image.fromarray(enhanced_image)

# Enhance contrast
enhancer = ImageEnhance.Contrast(pil_image)
enhanced_contrast_image = enhancer.enhance(2)  # Increase contrast level as needed

# Enhance sharpness
enhancer = ImageEnhance.Sharpness(enhanced_contrast_image)
final_image = enhancer.enhance(2)  # Increase sharpness level as needed

# Save the enhanced image
final_image.save('enhanced_character_image.png')

# Optionally, display the image using OpenCV
cv2.imshow('Enhanced Image', np.array(final_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
