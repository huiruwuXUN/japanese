# -------------------------------------------------------------------------------------------------------
# @author   Muhammad Arslan Amjad Qureshi         @Co-Author   Omair Soomro
# @date     2024-08-25
# @description This script creates the basic login page for Japanese Handwriting Analysis Project
# @description2 This script is now modified to create a Main Application Page, this Main Application Page
#               uploads the japanese handwriting leaflet image, is designed to process it(processing features
#               will be added later on and then give the results regarding this image.)
# --------------------------------------------------------------------------------------------------------

import tkinter as tk
from tkinter import messagebox, simpledialog
import sqlite3
import bcrypt

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
        hashed_password = result[0]  # hashed_password is already in bytes format from database
        if check_password(password, hashed_password):  # No need to encode the hashed_password
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


# Function to open the registration window
def register():
    username = simpledialog.askstring("Register", "Enter a new username:")
    password = simpledialog.askstring("Register", "Enter a new password:", show='*')

    if username and password:
        register_user(username, password)


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
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg *.jpeg *.png"), ("All files", ".*")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((image_display.winfo_width(), image_display.winfo_height()), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        image_display.config(image=img_tk)
        image_display.image = img_tk

# Function to reset the image display area
def reset_image():
    image_display.config(image='', text="Image Display Area")

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

root.mainloop()
