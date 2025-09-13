import onnxruntime as ort
import numpy as np
import tkinter as tk

# session = ort.InferenceSession("model.onnx")
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name

# input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
# result = session.run([output_name], {input_name: input_data})[0]

result = "HELLO WORLD"

root = tk.Tk()
root.title("ONNX Python App")
tk.Label(root, text=f"Prediction: {result}").pack()
root.mainloop()