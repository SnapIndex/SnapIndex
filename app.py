import onnxruntime as ort
import numpy as np
import tkinter as tk
import os, sys


if len(sys.argv) > 1:
    selected_folder = sys.argv[1]
else:
    selected_folder = "No folder selected"



root = tk.Tk()
root.title("SnapIndex")


tk.Label(root, text=f"Selected Folder:").pack()
tk.Label(root, text=selected_folder, wraplength=400, fg="blue").pack(padx=10, pady=5)


root.mainloop()