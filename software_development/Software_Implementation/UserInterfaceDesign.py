# -------------------------------------------------------------------------------------------------------
# @author   Muhammad Arslan Amjad Qureshi         @Co-Author   Omair Soomro
# @date     2024-08-25
# @description This script creates the basic login page for Japanese Handwriting Analysis Project
# @description2 This script is now modified to create a Main Application Page, this Main Application Page
#               uploads the japanese handwriting leaflet image, is designed to process it(processing features
#               will be added later on and then give the results regarding this image.)
# --------------------------------------------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk  # Import PIL for image handling


# Function to handle opening the main page after login
def open_main_page():
    # Hide or remove the login frame from view
    login_frame.pack_forget()
    # Call the function to initialize and display the main application page
    main_page()


# Function to set up and display the main application page
def main_page():

    global image_display  # Make image_display global to update it later

    # Create a new frame for the main application content
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Left panel setup for image list and upload button
    left_frame = tk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)
    upload_button = tk.Button(left_frame, text="Upload Image", command=open_file)
    upload_button.pack(pady=10)  # Add padding around the button for spacing
    image_listbox = tk.Listbox(left_frame)
    image_listbox.pack(fill=tk.BOTH, expand=True)

    # Main display area setup
    image_display = tk.Label(main_frame, text="Image Display Area", bg="grey")
    image_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Right panel setup for tools and options
    right_frame = tk.Frame(main_frame, width=200)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)
    process_button = tk.Button(right_frame, text="Process Image")
    process_button.pack(pady=10)
    options_frame = tk.LabelFrame(right_frame, text="Options")
    options_frame.pack(fill=tk.BOTH, expand=True)
    option1 = tk.Checkbutton(options_frame, text="Preprocessing")
    option1.pack(anchor='w')
    option2 = tk.Checkbutton(options_frame, text="Feature Extraction")
    option2.pack(anchor='w')
    option3 = tk.Checkbutton(options_frame, text="Clustering")
    option3.pack(anchor='w')

    # Bottom status bar to show status messages or progress
    status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Function to open a file dialog for image selection and display the selected image
def open_file():
    try:
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", ".jpg *.jpeg *.png"), ("All files", ".*")]
        )
        if file_path:  # If a file was selected
            print(f"File selected: {file_path}")
            # Load the image using PIL
            img = Image.open(file_path)
            # Resize the image to fit the display area while maintaining aspect ratio
            display_width = image_display.winfo_width()
            display_height = image_display.winfo_height()
            img.thumbnail((display_width, display_height), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img)
            # Update the image in the display area
            image_display.config(image=img_tk)
            image_display.image = img_tk  # Keep a reference to avoid garbage collection
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {e}")  # Correct usage of messagebox


# Initialize the main window for the application
root = tk.Tk() 
root.title("Japanese Handwriting Analysis Tool")
root.geometry("800x600")


login_frame = tk.Frame(root)
login_frame.pack(fill=tk.BOTH, expand=True)  # Pack the login frame to fill the window and allow expansion

login_label = tk.Label(login_frame, text="Login", font=("Arial", 16))  # Label for the login title
login_label.pack(pady=20)

username_label = tk.Label(login_frame, text="Username:")  # Label for the username entry
username_label.pack(pady=5)
username_entry = tk.Entry(login_frame)
username_entry.pack(pady=5)

password_label = tk.Label(login_frame, text="Password:")  # Label for the password entry
password_label.pack(pady=5)
password_entry = tk.Entry(login_frame, show="*")
password_entry.pack(pady=5)

login_button = tk.Button(login_frame, text="Login", command=open_main_page)
login_button.pack(pady=20)

root.mainloop()
