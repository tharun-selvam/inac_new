import pickle
import json
import sys

original_stdout = sys.stdout

file_name1 = "data/dataset/mountain_car/mixed/mm_run.pkl"
file_name2 = "data/dataset/mountain_car/expert/me_run.pkl"
acrobot_expert = "data/dataset/acrobot/expert/ae_run.pkl"
file_path = "test1.txt"

with open(file_name1, "rb") as f:
    dict = pickle.load(f)

with open(file_name2, "rb") as f:
    dict1 = pickle.load(f)


with open(acrobot_expert, "rb") as f:
    acro = pickle.load(f)

with open(file_path, "w") as f:
    sys.stdout = f
    print(dict)

sys.stdout = original_stdout

# print(dict)