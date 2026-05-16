import os
import pickle

DATA_DIR = r"train_augmented"
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

with open("class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)
print("Created class_names.pkl from dataset folders")
