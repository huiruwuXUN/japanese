import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Global variable to store list of images and current image index
image_list = []
current_image_index = 0

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

def open_main_page():
    root = tk.Tk()
    root.title("Japanese Handwriting Analysis Tool - Main Page")
    root.geometry("800x600")

    # Create Menu Bar
    menubar = tk.Menu(root)

    # Add File Menu
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)

    # Add Help Menu
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Help", command=show_help)
    help_menu.add_command(label="About", command=show_about)
    help_menu.add_command(label="Code References", command=show_code_references)
    menubar.add_cascade(label="Help", menu=help_menu)

    # Configure menu
    root.config(menu=menubar)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = tk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    upload_folder_button = tk.Button(left_frame, text="Upload Folder", command=lambda: open_folder(image_display))
    upload_folder_button.pack(pady=10)

    upload_image_button = tk.Button(left_frame, text="Upload Single Image", command=lambda: open_single_image(image_display))
    upload_image_button.pack(pady=10)

    image_listbox = tk.Listbox(left_frame)
    image_listbox.pack(fill=tk.BOTH, expand=True)

    global image_display
    image_display = tk.Label(main_frame, text="Image Display Area", bg="grey")
    image_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = tk.Frame(main_frame, width=200)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    process_button = tk.Button(right_frame, text="Process Image")
    process_button.pack(pady=10)

    reset_button = tk.Button(right_frame, text="Reset Image", command=lambda: reset_image(image_display))
    reset_button.pack(pady=10)

    next_button = tk.Button(right_frame, text="Next Image", command=lambda: next_image(image_display))
    next_button.pack(pady=10)

    prev_button = tk.Button(right_frame, text="Previous Image", command=lambda: previous_image(image_display))
    prev_button.pack(pady=10)

    options_frame = tk.LabelFrame(right_frame, text="Options")
    options_frame.pack(fill=tk.BOTH, expand=True)

    option1 = tk.Checkbutton(options_frame, text="Preprocessing")
    option1.pack(anchor='w')
    option2 = tk.Checkbutton(options_frame, text="Feature Extraction")
    option2.pack(anchor='w')
    option3 = tk.Checkbutton(options_frame, text="Clustering")
    option3.pack(anchor='w')

    status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()

# Function to open a folder and load all images inside
def open_folder(image_display):
    folder_path = filedialog.askdirectory()
    if folder_path:
        global image_list, current_image_index
        image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        current_image_index = 0
        if image_list:
            display_image(image_display, image_list[current_image_index])

# Function to open a single image
def open_single_image(image_display):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
    if file_path:
        global image_list, current_image_index
        image_list = [file_path]  # Treat single image as a list with one item
        current_image_index = 0
        display_image(image_display, image_list[current_image_index])

# Function to display an image on the image_display label
def display_image(image_display, image_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((image_display.winfo_width(), image_display.winfo_height()), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        image_display.config(image=img_tk, text="")
        image_display.image = img_tk  # Keep a reference to prevent garbage collection
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image:\n{e}")

# Function to reset the image display area
def reset_image(image_display):
    image_display.config(image='', text="Image Display Area")

# Function to show the next image in the folder
def next_image(image_display):
    global current_image_index
    if current_image_index < len(image_list) - 1:
        current_image_index += 1
        display_image(image_display, image_list[current_image_index])

# Function to show the previous image in the folder
def previous_image(image_display):
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
        display_image(image_display, image_list[current_image_index])
