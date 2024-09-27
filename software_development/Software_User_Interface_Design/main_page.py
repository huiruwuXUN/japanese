import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def show_help():
    help_text = """
    Japanese Handwriting Analysis Tool - Help Documentation
    
    1. Upload Image: Allows you to upload an image of Japanese handwriting.
    2. Process Image: This option will process the uploaded image (features to be added later).
    3. Reset Image: Resets the currently uploaded image.
    4. Preprocessing: Prepares the image for further analysis.
    5. Feature Extraction: Extracts key features from the handwriting for analysis.
    6. Clustering: Groups similar features from the handwriting images.

    For code documentation, please visit the project's repository or the codebase for more details.
    """
    messagebox.showinfo("Help & Documentation", help_text)

def open_main_page():
    root = tk.Tk()
    root.title("Japanese Handwriting Analysis Tool - Main Page")
    root.geometry("800x600")

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Add Help Button to the Main Page
    help_button = tk.Button(root, text="Help", command=show_help)
    help_button.pack(side=tk.TOP, pady=5)

    left_frame = tk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)
    
    upload_button = tk.Button(left_frame, text="Upload Image", command=lambda: open_file(image_display))
    upload_button.pack(pady=10)
    
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

def open_file(image_display):
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((image_display.winfo_width(), image_display.winfo_height()), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img)
            image_display.config(image=img_tk, text="")
            image_display.image = img_tk  # Keep a reference to prevent garbage collection
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

def reset_image(image_display):
    image_display.config(image='', text="Image Display Area")
