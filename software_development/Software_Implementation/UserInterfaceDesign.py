# ------------------------------------------------------------------------------------------
# @author   Muhammad Arslan Amjad Qureshi
# @date     2024-08-25
# @description This script Creates the basic login page for Japanese Handwriting Analysis Project
# ------------------------------------------------------------------------------------------


import tkinter as tk  # Import tkinter library for GUI creation


# Function to handle the login action
def open_main_page():
    # Placeholder function for transitioning to the main page
    print("Login successful!")  # For now, we'll just print a message indicating successful login


# Initialize the main window for the application
root = tk.Tk()  # Create the main window object
root.title("Japanese Handwriting Analysis")  # Set the window title to "Japanese Handwriting Analysis"
root.geometry("400x300")  # Set the window size to 400x300 pixels

# Create a frame for the login page
login_frame = tk.Frame(root)  # Create a frame widget to hold the login form elements
login_frame.pack(fill=tk.BOTH, expand=True)  # Pack the frame to fill the window and allow it to expand

# Create and pack the login page widgets
login_label = tk.Label(login_frame, text="Login", font=("Arial", 16))  # Create a label widget for the "Login" title
login_label.pack(pady=20)  # Pack the label into the frame with padding on the y-axis

username_label = tk.Label(login_frame, text="Username:")  # Create a label widget for the username field
username_label.pack(pady=5)  # Pack the label into the frame with some padding on the y-axis
username_entry = tk.Entry(login_frame)  # Create an entry widget for the user to input their username
username_entry.pack(pady=5)  # Pack the entry widget with some padding on the y-axis

password_label = tk.Label(login_frame, text="Password:")  # Create a label widget for the password field
password_label.pack(pady=5)  # Pack the label into the frame with some padding on the y-axis
password_entry = tk.Entry(login_frame, show="*")  # Create an entry widget for the password, hide input with '*'
password_entry.pack(pady=5)  # Pack the entry widget with some padding on the y-axis

login_button = tk.Button(login_frame, text="Login", command=open_main_page)  # Create a button widget to trigger login
login_button.pack(pady=20)  # Pack the button with padding on the y-axis

# Start the Tkinter event loop to run the application
root.mainloop()  # Enter the Tkinter event loop, waiting for user interaction
