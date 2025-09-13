import tkinter as tk
from tkinter import ttk

# Change main content when menu button is clicked
def show_content(name):
    for widget in content_frame.winfo_children():
        widget.destroy()
    lbl = tk.Label(content_frame, text=name, font=("Arial", 16))
    lbl.pack(pady=20)

root = tk.Tk()
root.title("SnapIndex")
root.geometry("900x500")
root.config(bg="white")

# Main layout (Sidebar + Content)
main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill="both", expand=True)

# Sidebar
sidebar = tk.Frame(main_frame, bg="white", width=200, relief="solid", bd=1)
sidebar.pack(side="left", fill="y")

# App title
title = tk.Label(sidebar, text="SnapIndex", font=("Arial", 14, "bold"), bg="white", anchor="w")
title.pack(padx=15, pady=20, anchor="w")

# Style for sidebar buttons
def make_sidebar_button(text, command, selected=False):
    bg_color = "#f5f7fa" if selected else "white"
    btn = tk.Button(
        sidebar, text=text, anchor="w",
        relief="flat", font=("Arial", 12),
        bg=bg_color, fg="black",
        activebackground="#e0e0e0", activeforeground="black",
        command=command
    )
    btn.pack(fill="x", pady=2, padx=5, ipady=8)
    return btn

# Sidebar buttons
btn1 = make_sidebar_button("🏠 Search", lambda: show_content("Search"))
btn2 = make_sidebar_button("📊 Organize", lambda: show_content("Organize"))
btn3 = make_sidebar_button("⚙️ Rename", lambda: show_content("Rename"), selected=True)

# Main content area
content_frame = tk.Frame(main_frame, bg="white")
content_frame.pack(side="right", fill="both", expand=True)

# Default content
show_content("Hello, welcome to SnapIndex!")

root.mainloop()
