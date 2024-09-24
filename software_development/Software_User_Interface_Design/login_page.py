import bcrypt
from tkinter import simpledialog, messagebox
import sqlite3

# Initialize SQLite connection
conn = sqlite3.connect('users.db')
c = conn.cursor()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        messagebox.showinfo("Registration", "User registered successfully!")
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Username already exists.")

def authenticate(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result:
        hashed_password = result[0]
        if check_password(password, hashed_password):
            return True
    return False

def login(username_entry, password_entry, root, open_main_page):
    username = username_entry.get()
    password = password_entry.get()
    
    if authenticate(username, password):
        messagebox.showinfo("Login", "Login Successful!")
        root.destroy()  # Close login window
        open_main_page()  # Open main application page
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

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
