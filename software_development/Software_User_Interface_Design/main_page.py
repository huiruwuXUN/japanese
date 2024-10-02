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
