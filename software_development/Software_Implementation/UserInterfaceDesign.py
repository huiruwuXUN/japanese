# -------------------------------------------------------------------------------------------------------
# @author   
#     Muhammad Arslan Amjad Qureshi         @Co-Author   Omair Soomro
# @date     
#     2024-08-25
# @description 
#     This script creates the basic login page for Japanese Handwriting Analysis Project
#     It has been modified to create a Main Application Page, allowing users to upload Japanese handwriting
#     images, process them (processing features to be added later), and display results.
# --------------------------------------------------------------------------------------------------------

import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
import sqlite3
import bcrypt
from PIL import Image, ImageTk

# Initialize SQLite connection
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create the users table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')
conn.commit()

# Function to hash a password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Function to check password
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Function to add a new user to the database
def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        messagebox.showinfo("Registration", "User registered successfully!")
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Username already exists.")

# Function to authenticate user on login
def authenticate(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result:
        hashed_password = result[0]
        if check_password(password, hashed_password):
            return True
    return False

# Function to handle login
def login():
    username = username_entry.get()
    password = password_entry.get()
    
    if authenticate(username, password):
        messagebox.showinfo("Login", "Login Successful!")
        open_main_page()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Function to open the registration window with improved validation
def register():
    while True:
        username = simpledialog.askstring("Register", "Enter a new username:")
        
        if username is None:
            break
        
        if not username:
            messagebox.showerror("Error", "Username field cannot be empty.")
            continue
        
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone() is not None:
            messagebox.showerror("Error", "Username already exists. Please choose a different username.")
            continue
        
        password = simpledialog.askstring("Register", "Enter a new password:", show='*')
        
        if password is None:
            break
        
        if not password:
            messagebox.showerror("Error", "Password field cannot be empty.")
            continue
        
        register_user(username, password)
        break

# Function to open the forgot password window
def forgot_password():
    username = simpledialog.askstring("Forgot Password", "Enter your username:")
    
    if username is None:
        return
    
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    
    if result:
        while True:
            new_password = simpledialog.askstring("Forgot Password", "Enter a new password:", show='*')
            
            if new_password is None:
                return
            
            if not new_password.strip():
                messagebox.showerror("Error", "Password field cannot be empty.")
                continue
            
            c.execute("UPDATE users SET password = ? WHERE username = ?", (hash_password(new_password), username))
            conn.commit()
            messagebox.showinfo("Password Reset", "Password reset successfully!")
            break
    else:
        messagebox.showerror("Error", "Username does not exist!")

# Function to handle opening the main page after login
def open_main_page():
    login_frame.pack_forget()
    main_page()

# Function to set up and display the main application page
def main_page():
    global image_display
    
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = tk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)
    
    upload_button = tk.Button(left_frame, text="Upload Image", command=open_file)
    upload_button.pack(pady=10)
    
    image_listbox = tk.Listbox(left_frame)
    image_listbox.pack(fill=tk.BOTH, expand=True)
    
    image_display = tk.Label(main_frame, text="Image Display Area", bg="grey")
    image_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    right_frame = tk.Frame(main_frame, width=200)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)
    
    process_button = tk.Button(right_frame, text="Process Image")
    process_button.pack(pady=10)
    
    reset_button = tk.Button(right_frame, text="Reset Image", command=reset_image)
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

# Function to open a file dialog for image selection
def open_file():
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

# Function to reset the image display area
def reset_image():
    image_display.config(image='', text="Image Display Area")

# Function to display help and documentation
def show_help():
    help_window = tk.Toplevel(root)
    help_window.title("Help & Documentation")
    help_window.geometry("500x400")
    
    # Create a notebook (tabbed view)
    notebook = ttk.Notebook(help_window)
    notebook.pack(expand=True, fill='both')
    
    # Create Overview Tab
    overview_tab = tk.Frame(notebook)
    notebook.add(overview_tab, text="Overview")
    overview_label = tk.Label(overview_tab, text="This application allows users to upload Japanese handwriting images...", wraplength=480)
    overview_label.pack(padx=10, pady=10)
    
    # Create How to Use Tab
    howto_tab = tk.Frame(notebook)
    notebook.add(howto_tab, text="How to Use")
    howto_label = tk.Label(howto_tab, text="1. Click on 'Upload Image' to select an image file.\n2. Use 'Process Image' to process the handwriting...", wraplength=480)
    howto_label.pack(padx=10, pady=10)
    
    # Create FAQ Tab
    faq_tab = tk.Frame(notebook)
    notebook.add(faq_tab, text="FAQ")
    faq_label = tk.Label(faq_tab, text="Q1: What image formats are supported?\nA: The application supports JPEG, PNG formats...", wraplength=480)
    faq_label.pack(padx=10, pady=10)
    
    # Create About Tab
    about_tab = tk.Frame(notebook)
    notebook.add(about_tab, text="About")
    about_label = tk.Label(about_tab, text="Developed by Muhammad Arslan Amjad Qureshi & Omair Soomro", wraplength=480)
    about_label.pack(padx=10, pady=10)

# Add a Help Menu to the existing main page
def add_help_menu(menu_bar):
    help_menu = tk.Menu(menu_bar, tearoff=0)
    help_menu.add_command(label="Help & Documentation", command=show_help)
    help_menu.add_separator()
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Developed by Muhammad Arslan Amjad Qureshi & Omair Soomro"))
    
    # Add the Help menu to the main menu bar
    menu_bar.add_cascade(label="Help", menu=help_menu)

# Initialize the main window
root = tk.Tk()
root.title("Japanese Handwriting Analysis Tool")
root.geometry("800x600")

login_frame = tk.Frame(root)
login_frame.pack(fill=tk.BOTH, expand=True)

login_label = tk.Label(login_frame, text="Login", font=("Arial", 16))
login_label.pack(pady=20)

username_label = tk.Label(login_frame, text="Username:")
username_label.pack(pady=5)
username_entry = tk.Entry(login_frame)
username_entry.pack(pady=5)

password_label = tk.Label(login_frame, text="Password:")
password_label.pack(pady=5)
password_entry = tk.Entry(login_frame, show="*")
password_entry.pack(pady=5)

login_button = tk.Button(login_frame, text="Login", command=login)
login_button.pack(pady=20)

register_button = tk.Button(login_frame, text="Register", command=register)
register_button.pack(pady=10)

forgot_password_button = tk.Button(login_frame, text="Forgot Password", command=forgot_password)
forgot_password_button.pack(pady=10)

# Bind Enter key to login
root.bind('<Return>', lambda event: login())

# Create a menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# Call the function to add help menu
add_help_menu(menu_bar)

root.mainloop()
