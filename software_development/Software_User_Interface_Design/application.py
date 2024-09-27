import tkinter as tk
from login_page import login, register, forgot_password
from main_page import open_main_page, show_help, show_about, show_code_references

def start_login_page():
    root = tk.Tk()
    root.title("Japanese Handwriting Analysis Tool - Login")
    root.geometry("400x300")

    # Add Menu Bar in Login Page
    menubar = tk.Menu(root)

    # Add Help Menu in the login page
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Help", command=show_help)
    help_menu.add_command(label="About", command=show_about)
    help_menu.add_command(label="Code References", command=show_code_references)
    menubar.add_cascade(label="Help", menu=help_menu)

    # Configure menu
    root.config(menu=menubar)

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
