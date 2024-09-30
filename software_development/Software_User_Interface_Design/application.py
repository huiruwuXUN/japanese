# -------------------------------------------------------------------------------------------------------
# @author
#     Muhammad Arslan Amjad Qureshi         @Co-Author   Omair Soomro
# @date
#     01-10-2024
# @description
#     This script handles the graphical user interface (GUI) for the login page of the Japanese
#     Handwriting Analysis Tool. The system allows users to log in, register, and recover forgotten
#     passwords using Tkinter. It also includes a menu bar providing help, about, and code references.
#     Upon successful login, users are redirected to the main page. This script interacts with other
#     modules such as login_page and main_page, which handle the backend logic for user authentication
#     and main application functionality.
# --------------------------------------------------------------------------------------------------------

import tkinter as tk
from login_page import login, register, forgot_password
from main_page import open_main_page, show_help, show_about, show_code_references


# Function to start and display the login page GUI
def start_login_page():
    # Create the main window for the login page
    root = tk.Tk()
    root.title("Japanese Handwriting Analysis Tool - Login")  # Set window title
    root.geometry("400x300")  # Set window dimensions

    # Add a menu bar to the login window
    menubar = tk.Menu(root)

    # Create a Help menu with three options: Help, About, Code References
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Help", command=show_help)  # Help command opens Help dialog
    help_menu.add_command(label="About", command=show_about)  # About command opens About dialog
    help_menu.add_command(label="Code References", command=show_code_references)  # Code References opens references section
    menubar.add_cascade(label="Help", menu=help_menu)  # Add Help menu to the menu bar

    # Configure the window to display the menu bar
    root.config(menu=menubar)

    # Create a frame to hold the login widgets (username, password, buttons)
    login_frame = tk.Frame(root)
    login_frame.pack(fill=tk.BOTH, expand=True)  # Make the frame expandable

    # Add a label for the login title
    login_label = tk.Label(login_frame, text="Login", font=("Arial", 16))
    login_label.pack(pady=20)  # Add padding around the label

    # Add username label and input entry
    username_label = tk.Label(login_frame, text="Username:")
    username_label.pack(pady=5)  # Add padding for spacing
    username_entry = tk.Entry(login_frame)  # Create a text entry for username input
    username_entry.pack(pady=5)

    # Add password label and input entry with hidden text (show="*")
    password_label = tk.Label(login_frame, text="Password:")
    password_label.pack(pady=5)
    password_entry = tk.Entry(login_frame, show="*")  # Password entry hides input characters
    password_entry.pack(pady=5)

    # Add login button, which calls the login function when clicked
    login_button = tk.Button(
        login_frame,
        text="Login",
        command=lambda: login(username_entry, password_entry, root, open_main_page)
    )
    login_button.pack(pady=20)

    # Add register button, which calls the register function when clicked
    register_button = tk.Button(login_frame, text="Register", command=register)
    register_button.pack(pady=10)

    # Add "Forgot Password" button, which calls the forgot_password function when clicked
    forgot_password_button = tk.Button(login_frame, text="Forgot Password", command=forgot_password)
    forgot_password_button.pack(pady=10)

    # Start the Tkinter main event loop, allowing the window to display and wait for user interaction
    root.mainloop()


# Main function check to start the program
if __name__ == "__main__":
    start_login_page()  # Call the function to launch the login page


####################### Future Work we can implement Further ###########################
# To enhance and expand the functionality of your Japanese Handwriting Analysis Tool,
# you could consider adding the following features and additional pages in the future:

'''
1) User Profile Page:
    1.1) Allow users to manage their profile, update personal information, and change passwords.
    1.2) Add features like viewing login history or updating security questions for better account security.

2) Admin Dashboard:
    2.1) Implement an admin page where administrators can manage users (e.g., view, edit, delete user accounts),
         view system analytics, and control access permissions.
    2.2) Track user activity, like login attempts, failed logins, and changes made in the system.

3) Handwriting Upload and Analysis Page:
    3.1) Create a dedicated page for users to upload images of Japanese handwriting.
    3.2) Implement functionality for image processing, Optical Character Recognition (OCR),
         and handwriting analysis directly within the application.

4) Progress Tracking and Reporting Page:
    4.1) Add a page that allows users to view reports on the handwriting analysis process.
    4.2) Include graphs and visualizations showing trends, accuracy, or improvements over time.

5)  User Notifications:
    5.1) Add a notifications page to alert users about changes in their account or updates on uploaded
         handwriting analysis.
    5.2) You can also notify users about password expiry, system updates, or new features added to the tool.

6) Multilingual Support:
    6.1) Add a settings page where users can switch between multiple languages
        (such as English and Japanese) for accessibility.

7) Help & Tutorials Page:
    7.1) Expand the help section to include video tutorials, FAQs, or detailed step-by-step guides.
    7.2) Implement an interactive guide to walk users through how to use various features of the tool effectively.

8) History & Archives Page:
    8.1) Allow users to access and manage previous analyses of handwriting they've uploaded,
         offering search and filter functionality to sort through past reports.

9) Collaboration and Sharing Features:
    9.1) Add a page where users can share their analysis results with other users or researchers.
    9.2) Implement collaborative features where multiple users can contribute or review specific
         handwriting samples for research purposes.

10) Data Export and Reporting:
    10.1) Provide an option for users to export their handwriting analysis data in various formats (PDF, CSV, etc.).
    10.2) Allow users to generate detailed reports based on the results of handwriting analyses.
'''
