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
from login_page import login, register, forgot_password
from main_page import open_main_page

def start_login_page():
    root = tk.Tk()
    root.title("Japanese Handwriting Analysis Tool - Login")
    root.geometry("400x300")

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

    login_button = tk.Button(login_frame, text="Login", command=lambda: login(username_entry, password_entry, root, open_main_page))
    login_button.pack(pady=20)

    register_button = tk.Button(login_frame, text="Register", command=register)
    register_button.pack(pady=10)

    forgot_password_button = tk.Button(login_frame, text="Forgot Password", command=forgot_password)
    forgot_password_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_login_page()
