import tkinter as tk
from tkinter import filedialog
import pickle
from MLEngine import MLModel

def load_model_encoder(model_path,label_encoder_path):
    model,le = None,None
    with open(model_path,"rb") as file:
        model = pickle.load(file)

    with open(label_encoder_path,'rb') as file:
        le = pickle.load(file)

    return model,le

def browse_file():
    filepath = filedialog.askopenfilename()
    if filepath:
        display.config(text="Selected File: {}".format(filepath))
        global selected_filepath
        selected_filepath = filepath

def search_file():
    if selected_filepath:
        print("Selected File:", selected_filepath)
        # Making decision
        model_path = "KNNClassifier.pickle"
        label_encoder_path = "labelEncoder.pickle"
        knn,le = load_model_encoder(model_path,label_encoder_path)
        ml_model = MLModel(knn,le,selected_filepath)
        ml_model.load_file()
        ml_model.modify_data_to_rows()
        pred = ml_model.predict()
        print(pred)
        if type(pred) == str:
            true_label = pred
            display.config(text = f"True Label : {true_label}")
        else:
            true_label = ml_model.encode_label(pred)
            display.config(text = f"True Label : {true_label[0]}")

    else:
        print("No file selected.")

root = tk.Tk()
root.title("Crop Recommendation")
root.geometry("500x500")
# Create a blank display
display = tk.Label(root, width=500, height=20, bg="white")
display.pack(pady=10)

# Browse button
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack(pady=5)

# Search button
search_button = tk.Button(root, text="Search", command=search_file)
search_button.pack(pady=5)

selected_filepath = None

root.mainloop()