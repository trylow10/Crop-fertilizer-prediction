from tkinter import *
import os
import tkinter
from PIL import ImageTk, Image
from tkinter import messagebox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
from mainApp import Main

from test_random_tree import RandomForest

root = Tk()
# root.geometry("400*200")
root.title("Predictlizer")
warnings.filterwarnings("ignore")


df = pd.read_csv("data/fertilizer.csv")
# print(df['Name'].unique())

df.dropna(how="any", inplace=True)

my_dict = dict(
    {
        "Stype": {"sandy": 1, "loamy": 2, "black": 3, "clayey": 4, "red": 5},
        "Ctype": {
            "maize": 4,
            "sugarcane": 9,
            "cotton": 13,
            "tobacco": 2,
            "paddy": 7,
            "barley": 1,
            "wheat": 11,
            "millets": 5,
            "oil seeds": 6,
            "pulses": 8,
            "ground Nut": 3,
        },
    }
)

labels = df["Stype"].astype("category").cat.categories.tolist()
replace1 = {"Stype": {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
# print(replace1)

features_replace = labels.copy()

features_replace = df.copy()

features_replace.replace(replace1, inplace=True)
# print(features_replace.head())

labels2 = df["Ctype"].astype("category").cat.categories.tolist()
replace2 = {"Ctype": {k: v for k, v in zip(labels2, list(range(1, len(labels2) + 1)))}}
# print(replace2)

features_replace.replace(replace2, inplace=True)
# print(features_replace.head())

data = features_replace.copy()
# data.head()

features = data.drop(columns=["Name"])
target = data["Name"]
labels = data["Name"]
y = data.round().groupby("Name").agg((["min", "max"]))
print(y)

x = pd.DataFrame(columns=data.columns[:-1])
db = df.round()

for label in data["Name"].unique():
    data_ = db[db["Name"] == label].iloc[:, :-1]

    for i, col in enumerate(x.columns):
        max_ = data_[col].max()
        min_ = data_[col].min()

        x.loc[label, col] = f"{min_} - {max_}"
print(x)


acc = []
model = []
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    features, target, test_size=0.2, random_state=2
)


def func_RF():
    # from sklearn.ensemble import RandomForestClassifier
    RF = RandomForest(n_trees=20, max_depth=4)
    RF.fit(Xtrain.values, Ytrain.values)

    predicted_values = RF.predict(Xtest.values)

    x = metrics.accuracy_score(Ytest, predicted_values)
    acc.append(x)
    model.append("RF")

    Temperature = nty_T.get()
    Humidity = nty_H.get()
    Moisture = nty_M.get()
    Soiltype = S_type.get()
    Croptype = C_type.get()
    Nitrogen = nty_N.get()
    Phosphorous = nty_P.get()
    Potassium = nty_K.get()

    Stype1 = my_dict["Stype"][Soiltype]
    Ctype1 = my_dict["Ctype"][Croptype]

    l = []
    l.append(Temperature)
    l.append(Humidity)
    l.append(Moisture)
    l.append(Ctype1)
    l.append(Stype1)
    l.append(Potassium)
    l.append(Phosphorous)
    l.append(Nitrogen)
    data = [l]
    data = [[int(item) for item in sublist] for sublist in data]

    result = RF.predict(data)

    Pdt_rf = Label(root, text=result, fg="Black")
    Pdt_rf.config(font=("Times", 18))
    Pdt_rf.grid(row=8, padx=10, column=3, sticky=W)

    acc_rf = Label(root, text=x * 100, fg="Black")
    acc_rf.config(font=("Times", 12))
    acc_rf.grid(row=9, padx=10, column=3, sticky=W)

    def rep_rf():
        report2 = classification_report(Ytest, predicted_values)
        messagebox.showinfo("RF Fertilizer prediction Report", report2)

    def conf_rf():
        y_pred = RF.predict(Xtest.values)
        return y_pred

    # print('y_pred :- ',result)

    cm = confusion_matrix(Ytest, conf_rf())
    display_labels = RF.get_classes()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    disp.plot(cmap=plt.get_cmap("Blues"))

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    rp_rf = Button(
        root, text="Report", command=rep_rf, bg="white", fg="Dark red", width=15
    )
    rp_rf.config(font=("Times new roman", 14))
    rp_rf.grid(row=8, column=4, padx=5, pady=10, sticky=W)

    conf_rf = Button(
        root,
        text="Confusion Matrix",
        command=conf_rf,
        bg="white",
        fg="Dark red",
        width=15,
    )
    conf_rf.config(font=("Times new roman", 14))
    conf_rf.grid(row=9, column=4, padx=5, pady=10, sticky=W)


def refresh():
    nty_T.delete(0, END)

    nty_H.delete(0, END)

    nty_M.delete(0, END)

    S_type.set("select")

    C_type.set("select")

    nty_N.delete(0, END)

    nty_P.delete(0, END)

    nty_K.delete(0, END)


def back():
    root.destroy()
    root1 = Tk()
    obj = Main(root1)
    root1.mainloop()


def validate(P):
    if not bool(P):
        return True
    if P.isdigit():
        if int(P) == 0 or int(P) <= 300:
            return True
        else:
            messagebox.showerror("showerror", "Value cannot be greater than 300")
            return False
    else:
        messagebox.showerror(
            "showerror", "It was not a Number. Please enter numeric value."
        )
        return False


head1 = Label(root, justify=LEFT, text="Predictlizer", fg="Dark green")
head1.config(
    font=(
        "Elephant",
        32,
    )
)
head1.grid(row=1, column=0, columnspan=12, padx=100)

head2 = Label(root, justify=LEFT, text="Fertilizer Predictor", fg="black")
head2.config(font=("Aharoni", 24, "bold"))
head2.grid(row=2, column=0, columnspan=12, padx=100)


lbl_T = Label(root, text="Temperature", fg="Black")
lbl_T.config(font=("Times", 18, "bold"))
lbl_T.grid(row=3, column=1, pady=10, padx=10, sticky=W)

lbl_H = Label(root, text="Humidity", fg="Black")
lbl_H.config(font=("Times", 18, "bold"))
lbl_H.grid(row=4, column=1, pady=10, padx=10, sticky=W)

lbl_M = Label(root, text="Moisture", fg="Black")
lbl_M.config(font=("Times", 18, "bold"))
lbl_M.grid(row=5, column=1, pady=10, padx=10, sticky=W)

lbl_S = Label(root, text="Soil Type", fg="Black")
lbl_S.config(font=("Times", 18, "bold"))
lbl_S.grid(row=6, column=1, pady=10, padx=10, sticky=W)

lbl_C = Label(root, text="Crop Type", fg="Black")
lbl_C.config(font=("Times", 18, "bold"))
lbl_C.grid(row=7, column=1, pady=10, padx=10, sticky=W)

lbl_N = Label(root, text="Nitrogen", fg="Black")
lbl_N.config(font=("Times", 18, "bold"))
lbl_N.grid(row=8, column=1, pady=10, padx=10, sticky=W)

lbl_P = Label(root, text="Phosphorous", fg="Black")
lbl_P.config(font=("Times", 18, "bold"))
lbl_P.grid(row=9, column=1, pady=10, padx=10, sticky=W)

lbl_K = Label(root, text="Potassium", fg="Black")
lbl_K.config(font=("Times", 18, "bold"))
lbl_K.grid(row=10, column=1, pady=10, padx=10, sticky=W)

Crops = [
    "maize",
    "sugarcane",
    "cotton",
    "tobacco",
    "paddy",
    "barley",
    "wheat",
    "millets",
    "oil seeds",
    "pulses",
    "ground Nut",
]
Soils = ["sandy", "loamy", "black", "clayey", "red"]
S_type = StringVar()
S_type.set("select")
C_type = StringVar()
C_type.set("select")


nty_T = Entry(root, validate="key", validatecommand=(root.register(validate), "%P"))
nty_T.grid(row=3, column=2, padx=10, sticky=W)

nty_H = Entry(root, validate="key", validatecommand=(root.register(validate), "%P"))
nty_H.grid(row=4, column=2, padx=10, sticky=W)

nty_M = Entry(root, validate="key", validatecommand=(root.register(validate), "%P"))
nty_M.grid(row=5, column=2, padx=10, sticky=W)

nty_S = OptionMenu(root, S_type, *Soils)
nty_S.grid(row=6, column=2, padx=10, sticky=W)

nty_C = OptionMenu(root, C_type, *Crops)
nty_C.grid(row=7, column=2, padx=10, sticky=W)

nty_N = Entry(root, validate="key", validatecommand=(root.register(validate), "%P"))
nty_N.grid(row=8, column=2, padx=10, sticky=W)

nty_P = Entry(root, validate="key", validatecommand=(root.register(validate), "%P"))
nty_P.grid(row=9, column=2, padx=10, sticky=W)

nty_K = Entry(root, validate="key", validatecommand=(root.register(validate), "%P"))
nty_K.grid(row=10, column=2, padx=10, sticky=W)


lr = Button(
    root,
    text="Predict using RF",
    command=func_RF,
    bg="white",
    fg="Dark red",
    width=15,
    padx=80,
)
lr.config(font=("Times new roman", 16))
lr.grid(row=7, column=3, columnspan=6, padx=10, pady=10, sticky=W)

ref = Button(
    root, text="Refresh", command=refresh, bg="grey", fg="Dark green", width=10
)
ref.config(font=("Times new roman", 16))
ref.grid(row=10, column=3, padx=10, pady=10, sticky=W)

bck = Button(root, text="Go back", command=back, bg="grey", fg="Dark green", width=10)
bck.config(font=("Times new roman", 16))
bck.grid(row=10, column=4, padx=10, pady=10, sticky=W)


root.mainloop()
