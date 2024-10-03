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


############################### Recommendations for the Future Work #####################################
'''
1. Improved User Interface (UI) Design

    Responsive Layout: Make the UI responsive and visually appealing by implementing a modern, polished layout using a combination of Tkinter widgets and styles (you can use ttk widgets for a more modern look).
    Custom Themes: Apply custom themes or styles using libraries like ttkbootstrap to give the interface a more professional appearance, with customizable fonts, colors, and buttons.
    Form Validation: Add form validation for username and password inputs, displaying user-friendly error messages in real-time (e.g., when the password is too short or the username contains invalid characters).
    Login Animation/Transitions: Add subtle animations or transitions (e.g., fading effects, button hover effects) to give the login interface a smoother and more engaging feel.

2. Security Enhancements

    Brute Force Protection: Implement a mechanism to prevent brute force attacks by limiting the number of failed login attempts. After a certain number of failed attempts, lock the account temporarily or require CAPTCHA verification.
    CAPTCHA Integration: Add CAPTCHA to the login page to prevent bots from attempting unauthorized access.
    Two-Factor Authentication (2FA): Add 2FA to enhance security by requiring users to provide an additional verification code sent to their email or mobile phone after entering their password.
    Password Strength Meter: Include a password strength indicator when registering or resetting passwords to guide users in creating strong, secure passwords.

3. Password Management Improvements

    Password Recovery via Email: Allow users to reset their passwords via a secure email link. When users forget their password, they can request a reset link that is sent to their registered email address. This adds a more professional touch and enhances security.
    Encrypted Password Storage: Implement more advanced password encryption algorithms and database security features to ensure that sensitive information, such as usernames and passwords, is stored securely.
    Password Expiry: Enforce periodic password expiration, prompting users to update their passwords after a certain period (e.g., 90 days), with reminders sent via email.

4. User Experience (UX) Enhancements

    Remember Me Feature: Add a "Remember Me" checkbox that allows users to stay logged in on their device until they manually log out. This can be implemented using secure session management or tokens.
    Login Feedback: Provide better feedback during login processes, such as a loading spinner or progress bar to show the system is authenticating the user.
    Forgot Username: In addition to resetting passwords, provide an option for users to recover their forgotten username via email or phone number.

5. User Account Management

    Profile Management: After logging in, allow users to manage their profiles, where they can update personal information, change passwords, or review account activity (e.g., last login, recent changes).
    Role-Based Access Control (RBAC): Introduce role-based access control, where different users (e.g., admin, regular user) have different access levels to the system’s features. This will add versatility to the login system.

6. Database and Backend Enhancements

    Database Schema Optimization: Consider adding more detailed user attributes to the SQLite database (e.g., email, phone number, last login time, password reset date) to improve account management and traceability.
    Migrate to a More Scalable Database: Move from SQLite to a more scalable database like PostgreSQL or MySQL if the application needs to support a larger number of users.
    Session Management: Implement session handling (especially useful if converting the tool into a web-based application). This will ensure that users remain logged in for a period and allow easy logout from multiple devices.

7. Accessibility Features

    Accessibility Compliance: Ensure the login page is accessible to all users, including those with disabilities. This can be achieved by adding keyboard navigation support, screen reader compatibility, and proper color contrast for better visibility.
    Language Support: Add multilingual support to accommodate users from different regions by allowing the interface to switch between languages (e.g., English, Japanese).

8. Integration with External Services

    Social Login Options: Allow users to sign up and log in using their social media accounts (e.g., Google, Facebook, LinkedIn) to streamline the login process and improve convenience.
    OAuth 2.0 Integration: Implement OAuth 2.0 protocol for secure and scalable login using third-party services (such as Google accounts), adding more flexibility for the login process.

9. Logging and Monitoring

    Login Activity Logging: Log all login attempts, both successful and failed, along with timestamps and IP addresses. This provides valuable insights for system administrators and can help detect suspicious activity.
    Audit Trail: Keep an audit trail of user activities such as account creation, password reset, and profile changes. This improves security and accountability.

10. Testing and Scalability

    Unit and Integration Testing: Implement thorough testing for all login processes, including unit tests for password hashing, authentication, and edge cases (e.g., failed login attempts). This ensures reliability and robustness of the login system.
    Scalability: Prepare the login system for scalability by considering the use of load balancing and performance optimization techniques if the tool becomes widely used.

'''
