# -------------------------------------------------------------------------------------------------------
# @author   
#     Muhammad Arslan Amjad Qureshi         @Co-Author   Omair Soomro
# @date     
#     2024-08-25
# @description 
#     This script handles user registration, authentication, and password management for a simple 
#     login system using bcrypt for password hashing and SQLite for storing user data.
# --------------------------------------------------------------------------------------------------------

import bcrypt  # Importing bcrypt library for password hashing and verification
from tkinter import simpledialog, messagebox  # Importing GUI components from tkinter for dialogs and messages
import sqlite3  # Importing sqlite3 library to manage database operations

# Initialize SQLite connection to 'users.db'
# Creates the database file if it doesn't exist
conn = sqlite3.connect('users.db')
c = conn.cursor()  # Create a cursor to interact with the SQLite database

# Function to hash the user's password using bcrypt
def hash_password(password):
    """
    Hash the password using bcrypt with a randomly generated salt.
    :param password: The plain-text password entered by the user
    :return: The hashed password
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Function to check if the entered password matches the hashed password in the database
def check_password(password, hashed):
    """
    Check if the provided password matches the stored hashed password.
    :param password: The plain-text password entered by the user
    :param hashed: The hashed password retrieved from the database
    :return: True if the password matches, otherwise False
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Function to register a new user in the SQLite database
def register_user(username, password):
    """
    Registers a new user by saving their username and hashed password into the database.
    If the username already exists, it raises an error.
    :param username: The username entered by the user
    :param password: The plain-text password to be hashed and stored
    """
    try:
        # Insert username and hashed password into the database
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()  # Commit the changes to the database
        messagebox.showinfo("Registration", "User registered successfully!")
    except sqlite3.IntegrityError:  # If username already exists, raise an error
        messagebox.showerror("Error", "Username already exists.")

# Function to authenticate a user during login
def authenticate(username, password):
    """
    Authenticate the user by comparing the entered password with the hashed password in the database.
    :param username: The username entered by the user
    :param password: The plain-text password entered by the user
    :return: True if authentication is successful, otherwise False
    """
    # Retrieve the hashed password for the entered username
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result:
        # If username exists, check if the entered password matches the stored hash
        hashed_password = result[0]
        if check_password(password, hashed_password):
            return True  # Authentication successful
    return False  # Authentication failed

# Function to handle user login process
def login(username_entry, password_entry, root, open_main_page):
    """
    Handle the login process by checking the entered username and password.
    If authentication succeeds, close the login window and open the main application page.
    :param username_entry: The Entry widget containing the entered username
    :param password_entry: The Entry widget containing the entered password
    :param root: The root window (login window) to be destroyed upon successful login
    :param open_main_page: Function to open the main application page
    """
    username = username_entry.get()
    password = password_entry.get()
    
    if authenticate(username, password):
        messagebox.showinfo("Login", "Login Successful!")
        root.destroy()  # Close the login window
        open_main_page()  # Open the main application page
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Function to handle user registration
def register():
    """
    Register a new user by prompting them for a username and password.
    The system ensures that the username is unique and that both fields are non-empty.
    """
    while True:
        # Prompt the user for a new username
        username = simpledialog.askstring("Register", "Enter a new username:")
        
        if username is None:  # If the user cancels, exit the registration process
            break
        
        if not username:  # If username is empty, show an error
            messagebox.showerror("Error", "Username field cannot be empty.")
            continue
        
        # Check if the username already exists in the database
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone() is not None:
            messagebox.showerror("Error", "Username already exists. Please choose a different username.")
            continue
        
        # Prompt the user for a password
        password = simpledialog.askstring("Register", "Enter a new password:", show='*')
        
        if password is None:  # If the user cancels, exit the registration process
            break
        
        if not password:  # If password is empty, show an error
            messagebox.showerror("Error", "Password field cannot be empty.")
            continue
        
        # Register the user with the provided username and password
        register_user(username, password)
        break  # Exit the loop after successful registration

# Function to handle password reset if the user forgot their password
def forgot_password():
    """
    Reset the user's password if they forgot it by verifying the username.
    The user is then prompted to enter a new password, which is updated in the database.
    """
    # Prompt the user for their username
    username = simpledialog.askstring("Forgot Password", "Enter your username:")
    
    if username is None:  # If the user cancels, exit the process
        return
    
    # Check if the username exists in the database
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    
    if result:
        # If username exists, prompt for a new password
        while True:
            new_password = simpledialog.askstring("Forgot Password", "Enter a new password:", show='*')
            
            if new_password is None:  # If the user cancels, exit the process
                return
            
            if not new_password.strip():  # If the new password is empty, show an error
                messagebox.showerror("Error", "Password field cannot be empty.")
                continue
            
            # Update the password for the provided username
            c.execute("UPDATE users SET password = ? WHERE username = ?", (hash_password(new_password), username))
            conn.commit()  # Commit the changes to the database
            messagebox.showinfo("Password Reset", "Password reset successfully!")
            break  # Exit the loop after successful password reset
    else:
        # Show an error if the username does not exist in the database
        messagebox.showerror("Error", "Username does not exist!")
