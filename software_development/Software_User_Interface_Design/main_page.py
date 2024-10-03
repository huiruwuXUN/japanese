# -------------------------------------------------------------------------------
# @author
#    Muhammad Arslan Amjad Qureshi       @Co-Author      Omair Soomro
# @date
#    01-10-2024
# @description
#    This script handles the main page for the Japanese Handwriting Analysis Tool. 
#    It allows users to upload images or folders of images, and provides functionality 
#    for viewing, navigating, resetting, and processing the images. The main page includes 
#    options such as preprocessing, feature extraction, and clustering for image analysis, 
#    along with help and code reference menus. It is designed for analyzing Japanese 
#    handwriting from WWII-era leaflets using Tkinter as the GUI framework.
# -------------------------------------------------------------------------------

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Global variables to store a list of images and the current image index
image_list = []  # List to hold paths of images
current_image_index = 0  # Tracks the index of the currently displayed image

# Function to display help documentation in a messagebox
def show_help():
    help_text = """
    Japanese Handwriting Analysis Tool - Help Documentation
    
    1. Upload Folder: Allows you to upload a folder of images of Japanese handwriting.
    2. Upload Single Image: Allows you to upload a single image (PNG or JPG).
    3. Process Image: This option will process the currently displayed image (features to be added later).
    4. Reset Image: Resets the currently displayed image.
    5. Preprocessing: Prepares the image for further analysis.
    6. Feature Extraction: Extracts key features from the handwriting for analysis.
    7. Clustering: Groups similar features from the handwriting images.

    For code documentation, please visit the project's repository or the codebase for more details.
    """
    messagebox.showinfo("Help & Documentation", help_text)

# Function to display about information in a messagebox
def show_about():
    about_text = """
    Japanese Handwriting Analysis Tool
    
    Version: 1.0
    Developed by: Muhammad Arslan Amjad Qureshi, Omair Soomro
    Date: 2024-08-25

    This application is developed for analyzing Japanese handwriting from WWII-era leaflets. 
    It provides tools for image upload, preprocessing, feature extraction, and clustering.
    """
    messagebox.showinfo("About", about_text)

# Function to display code references used in the project
def show_code_references():
    code_ref_text = """
    Code References:
    
    1. Bcrypt Library: Used for secure password hashing.
    2. Tkinter Library: Used for the graphical user interface.
    3. PIL (Pillow): Used for image processing and display.
    4. SQLite: Database used for user registration and authentication.
    
    For more detailed references, please visit the project's repository.
    """
    messagebox.showinfo("Code References", code_ref_text)

# Main function that sets up the GUI for the main page
def open_main_page():
    root = tk.Tk()  # Create a Tkinter root window
    root.title("Japanese Handwriting Analysis Tool - Main Page")  # Set the title of the window
    root.geometry("800x600")  # Set the default window size

    # Create Menu Bar
    menubar = tk.Menu(root)

    # Add File Menu with Exit option
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Exit", command=root.quit)  # Command to quit the application
    menubar.add_cascade(label="File", menu=file_menu)  # Add File menu to menubar

    # Add Help Menu with Help, About, and Code References options
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Help", command=show_help)  # Show help documentation
    help_menu.add_command(label="About", command=show_about)  # Show about information
    help_menu.add_command(label="Code References", command=show_code_references)  # Show code references
    menubar.add_cascade(label="Help", menu=help_menu)  # Add Help menu to menubar

    # Configure the menu bar in the root window
    root.config(menu=menubar)

    # Main frame to hold all widgets
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)  # Expand frame to fill the window

    # Left frame for the upload options
    left_frame = tk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    # Button to upload a folder of images
    upload_folder_button = tk.Button(left_frame, text="Upload Folder", command=lambda: open_folder(image_display))
    upload_folder_button.pack(pady=10)  # Add padding for spacing

    # Button to upload a single image
    upload_image_button = tk.Button(left_frame, text="Upload Single Image", command=lambda: open_single_image(image_display))
    upload_image_button.pack(pady=10)  # Add padding for spacing

    # Listbox to display the uploaded image filenames
    image_listbox = tk.Listbox(left_frame)
    image_listbox.pack(fill=tk.BOTH, expand=True)

    # Main area to display the image
    global image_display
    image_display = tk.Label(main_frame, text="Image Display Area", bg="grey")  # Placeholder area for displaying images
    image_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Right frame for image manipulation buttons and options
    right_frame = tk.Frame(main_frame, width=200)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Button for processing images (feature not implemented yet)
    process_button = tk.Button(right_frame, text="Process Image")
    process_button.pack(pady=10)

    # Button to reset the displayed image
    reset_button = tk.Button(right_frame, text="Reset Image", command=lambda: reset_image(image_display))
    reset_button.pack(pady=10)

    # Button to display the next image in the list
    next_button = tk.Button(right_frame, text="Next Image", command=lambda: next_image(image_display))
    next_button.pack(pady=10)

    # Button to display the previous image in the list
    prev_button = tk.Button(right_frame, text="Previous Image", command=lambda: previous_image(image_display))
    prev_button.pack(pady=10)

    # Label frame for additional image processing options
    options_frame = tk.LabelFrame(right_frame, text="Options")
    options_frame.pack(fill=tk.BOTH, expand=True)

    # Checkbuttons for different processing options (Preprocessing, Feature Extraction, Clustering)
    option1 = tk.Checkbutton(options_frame, text="Preprocessing")
    option1.pack(anchor='w')
    option2 = tk.Checkbutton(options_frame, text="Feature Extraction")
    option2.pack(anchor='w')
    option3 = tk.Checkbutton(options_frame, text="Clustering")
    option3.pack(anchor='w')

    # Status bar at the bottom of the window
    status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # Start the Tkinter main loop
    root.mainloop()

# Function to open a folder and load all images inside
def open_folder(image_display):
    folder_path = filedialog.askdirectory()  # Open a dialog to select a directory
    if folder_path:
        global image_list, current_image_index
        # Get all image files from the folder
        image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        current_image_index = 0  # Reset index to the first image
        if image_list:
            display_image(image_display, image_list[current_image_index])  # Display the first image

# Function to open a single image file
def open_single_image(image_display):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
    if file_path:
        global image_list, current_image_index
        image_list = [file_path]  # Treat single image as a list with one item
        current_image_index = 0
        display_image(image_display, image_list[current_image_index])  # Display the selected image

# Function to display an image in the image_display area
def display_image(image_display, image_path):
    try:
        img = Image.open(image_path)  # Open the image file
        img.thumbnail((image_display.winfo_width(), image_display.winfo_height()), Image.ANTIALIAS)  # Resize the image
        img_tk = ImageTk.PhotoImage(img)  # Convert to a format Tkinter can display
        image_display.config(image=img_tk, text="")  # Display the image
        image_display.image = img_tk  # Keep a reference to avoid garbage collection
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image:\n{e}")  # Show error message if image fails to open

# Function to reset the image display area
def reset_image(image_display):
    image_display.config(image='', text="Image Display Area")  # Clear the image and reset to default text

# Function to show the next image in the list
def next_image(image_display):
    global current_image_index
    if current_image_index < len(image_list) - 1:  # Check if the current image is not the last
        current_image_index += 1  # Increment the image index
        display_image(image_display, image_list[current_image_index])  # Display the next image

# Function to show the previous image in the list
def previous_image(image_display):
    global current_image_index
    if current_image_index > 0:  # Check if the current image is not the first
        current_image_index -= 1  # Decrement the image index
        display_image(image_display, image_list[current_image_index])  # Display the previous image


############  Recommendations for the Future Work for the Main Application Page #############
'''
-----> Recommendation for the Future Work or the Future Enchancements integrated into our main_page.py or software.
1. Image Processing Features

    Image Preprocessing: Expand the preprocessing feature by allowing users to apply common image processing techniques such as contrast adjustment, noise removal, binarization (thresholding), and resizing. These will help clean up the scanned images for better analysis.
        Example: cv2 library integration for preprocessing filters such as cv2.GaussianBlur(), cv2.adaptiveThreshold(), etc.

    Feature Extraction Automation: Add automatic feature extraction functionality where the tool could detect and highlight key handwriting features (e.g., stroke density, slant, character spacing). This can assist in comparing handwriting characteristics.
        Use libraries such as OpenCV or implement custom algorithms for feature extraction.

    Optical Character Recognition (OCR): Integrate an OCR system to recognize characters from the uploaded images. This would be useful for identifying characters, words, or symbols in the handwriting.
        Googles Tesseract OCR could be an option here (pytesseract).

2. Handwriting Comparison

    Comparative Analysis: Allow the system to compare multiple handwriting samples to find similarities or differences between the authors. This could involve clustering techniques and similarity metrics to identify different handwriting styles.
        Use clustering algorithms such as K-Means, Agglomerative Clustering, or DBSCAN to group similar features from multiple images.

    Authorship Identification: Implement a system where the tool attempts to identify potential authorship by analyzing and matching the key features in different handwriting samples. This could be especially useful for the WWII-era leaflet project.

3. Improved User Interface

    Image Zoom & Pan Functionality: Add the ability to zoom in and out of the displayed images, as well as pan to navigate through different parts of the image. This will allow users to closely inspect smaller details in the handwriting.
        You could implement zoom functionality using PIL.ImageOps.scale() and mouse events for panning.

    Image Annotation: Add tools that allow users to annotate parts of the image, mark features, or highlight areas of interest. This can be useful for further analysis and user collaboration.
        You can use a canvas widget to allow drawing directly over the image.

    Batch Processing: Allow batch processing of images for faster analysis. Users can upload a folder of images and apply preprocessing, feature extraction, and clustering operations to all of them simultaneously.

4. Integration with Machine Learning Models

    Handwriting Classification: Integrate a machine learning model to classify handwriting based on specific parameters (e.g., slant, curvature, spacing, etc.). This could be used to classify the handwriting into predefined categories, such as style or period.
        A simple model can be built using scikit-learn or more advanced neural networks like CNNs (Convolutional Neural Networks).

    Real-time Feedback: Provide real-time feedback as images are processed, showing immediate results of preprocessing and feature extraction without requiring the user to manually check the image after each step.

5. Version Control for Image Processing

    Undo/Redo Functionality: Add the ability for users to undo and redo changes they have made during image processing. This can be useful when working with multiple transformations, so users can easily revert back to previous states.
        Maintain a history of transformations applied to the images, and allow rollback or reapplication of any step.

    Save Progress: Allow users to save their current analysis progress, including annotations, feature extraction results, and clustering. This way, users can pick up where they left off when returning to the tool.

6. User Management and Authentication

    User Authentication and Role Management: Expand the user registration and login system to support role-based access control. Different users (e.g., researchers, admins) may have different levels of access to the tool’s features.
        Admins might have the ability to manage data, add new users, or fine-tune system settings, while researchers may focus on analysis tasks.

7. Data Export and Report Generation

    Export Results: Enable users to export the analysis results, including processed images, extracted features, and comparative analysis reports, in common formats such as CSV, PDF, or JSON.
        This will allow the researchers to further study or share their findings.

    Automated Report Generation: Build a system to automatically generate detailed reports based on the analysis, including statistics about handwriting features, images, comparisons, and the overall conclusion of the analysis.
        This feature can be helpful for documenting findings related to WWII-era leaflet authorship or handwriting patterns.

8. Enhanced Data Management

    Image Metadata: Store and display metadata about each image (e.g., filename, upload date, image resolution, etc.) to give users more context about the images they are working with.
        You can use SQLite or a lightweight database to manage the image metadata.

    Tagging and Labeling System: Introduce the ability to tag or label images with important information, such as the writer’s identity (if known), date, or the specific content of the leaflets. This can help categorize the images for easier future access and retrieval.

9. Collaboration Features

    Multi-user Collaboration: Allow multiple users to collaborate on the same project by enabling shared access to image collections, annotations, and analysis results. Users can leave comments or suggestions for others to view and respond to.
        Integration with cloud storage or shared databases could make this feature possible.

    Version History for Images: Implement version control for images, so any changes made during processing or annotation are logged. This will allow users to view the evolution of the analysis or restore a previous version of an image.

10. Cloud Integration and Scalability

    Cloud Storage Integration: Allow users to upload and retrieve images from cloud storage (e.g., Google Drive, AWS S3) to make it easier to work with large datasets. This will also enable collaboration among remote teams.
        Cloud-based image processing pipelines can also help scale the tool for handling large sets of high-resolution images.

11. Mobile Version or Web Application

    Web-based Application: Consider converting the tool into a web-based application using a framework like Flask or Django. This could provide easier access for users and allow cloud-based storage and analysis.
        A web-based interface could make it more accessible for international collaboration, especially for researchers who may want to access the tool remotely.
'''
