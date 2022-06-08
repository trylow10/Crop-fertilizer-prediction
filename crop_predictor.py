

from tkinter import *
import os
# from gui_stuff import *
from PIL import ImageTk,Image
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings

from mainApp import Main

root = Tk()
# root.geometry("400*200")
root.title("Predictlizer")
warnings.filterwarnings('ignore')


df=pd.read_csv('data/cp1.csv')
#print(df.head())
#print(df.tail())
#print(df['label'].unique())

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']
y = df.round().groupby('label').agg((['min', 'max']))
#print(y)

x = pd.DataFrame(columns=df.columns[:-1])
db = df.round()

for label in df['label'].unique():
    data_ = db[db['label'] == label].iloc[:,:-1]
    
    for i, col in enumerate(x.columns):
        max_ = data_[col].max()
        min_ = data_[col].min()

        x.loc[label,col] = f'{min_} - {max_}'
print(x)


acc = []
model = []
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)



def func_RF():
    from sklearn.ensemble import RandomForestClassifier
    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)

    predicted_values = RF.predict(Xtest)

    x = metrics.accuracy_score(Ytest, predicted_values)
    acc.append(x)
    model.append('RF')

    N = nty_N.get()
    P = nty_P.get()
    K = nty_K.get()                 
    Temperature = nty_T.get()
    Humidity = nty_H.get()
    ph = nty_Ph.get()
    Rainfall = nty_R.get()

    l=[]
    l.append(N)
    l.append(P)
    l.append(K)
    l.append(Temperature)
    l.append(Humidity)
    l.append(ph)
    l.append(Rainfall)
    data=[l]

    prediction = RF.predict(data)
    Pdt_rf = Label(root, text=prediction,fg="Black" )
    Pdt_rf.config(font=("Times", 18))
    Pdt_rf.grid(row=8, padx=10, column=3,sticky=W)

    acc_rf = Label(root, text= x*100,fg="Black" )
    acc_rf.config(font=("Times", 12))
    acc_rf.grid(row=9, padx=10, column=3,sticky=W)

        
    def conf_rf():
            plot_confusion_matrix(RF, Xtest, Ytest, cmap = plt.get_cmap('Blues'))  
            plt.show()

    def rep_rf():
        report2 = classification_report(Ytest,predicted_values)
        messagebox.showinfo("RF Crop Prediction Report", report2)

    rp_rf = Button(root, text="Report", command=rep_rf,bg="white",fg="Dark red", width=15)
    rp_rf.config(font=("Times new roman", 14))
    rp_rf.grid(row=8, column=4,padx=5,pady=10, sticky=W)

    conf_rf = Button(root, text="Confusion Matrix", command=conf_rf,bg="white",fg="Dark red", width=15)
    conf_rf.config(font=("Times new roman", 14))
    conf_rf.grid(row=9, column=4,padx=5,pady=10, sticky=W)



def refresh():
    nty_N.delete(0,END)
    nty_P.delete(0,END)
    nty_T.delete(0,END)
    nty_K.delete(0,END)
    nty_Ph.delete(0,END)
    nty_R.delete(0,END)
    nty_H.delete(0,END) 
    nty_K.delete(0,END)

def back():
    root.destroy()
    root2 =Tk()
    obj = Main(root2)
    root2.mainloop()
    

def validate(P):       
    if P.isdigit():
        if int(P)==0 or int(P)<=300:
            return True
        else:
            messagebox.showerror("showerror", "Value cannot be greater than 300")
            return False
    else:
        # messagebox.showerror("showerror", "It was not a Number. Please enter numeric value.")
        return True   


head1 = Label(root, justify=LEFT, text="Predictlizer", fg="Dark green" )
head1.config(font=("Elephant", 32,))
head1.grid(row=1, column=0, columnspan=12, padx=100)
  
head2 = Label(root, justify=LEFT, text="Crop Predictor", fg="black" )
head2.config(font=("Aharoni", 22))
head2.grid(row=2, column=0, columnspan=12, padx=100)


lbl_N = Label(root, text="Nitrogen",fg="Black" )
lbl_N.config(font=("Times", 18,"bold"))
lbl_N.grid(row=4, column=1, pady=10, padx=10, sticky=W)

lbl_P = Label(root, text="Phosphorous",fg="Black" )
lbl_P.config(font=("Times", 18,"bold"))
lbl_P.grid(row=5, column=1, pady=10, padx=10, sticky=W)

lbl_K = Label(root, text="Potassium",fg="Black" )
lbl_K.config(font=("Times", 18,"bold"))
lbl_K.grid(row=6, column=1, pady=10, padx=10, sticky=W)

lbl_T = Label(root, text="Temperature",fg="Black" )
lbl_T.config(font=("Times", 18,"bold"))
lbl_T.grid(row=7, column=1, pady=10, padx=10, sticky=W)

lbl_H = Label(root, text="Humidity", fg="Black")
lbl_H.config(font=("Times", 18,"bold"))
lbl_H.grid(row=8, column=1, pady=10, padx=10, sticky=W)

lbl_Ph = Label(root, text="Ph", fg="Black")
lbl_Ph.config(font=("Times", 18,"bold"))
lbl_Ph.grid(row=9, column=1, pady=10, padx=10, sticky=W)

lbl_R = Label(root, text="Rainfall", fg="Black")
lbl_R.config(font=("Times", 18,"bold"))
lbl_R.grid(row=10, column=1, pady=10, padx=10, sticky=W)


nty_N = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_N.grid(row=4, column=2,padx=10,sticky=W)

nty_P = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_P.grid(row=5, column=2,padx=10,sticky=W)

nty_K = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_K.grid(row=6, column=2,padx=10,sticky=W)

nty_T = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_T.grid(row=7, column=2,padx=10,sticky=W)

nty_H = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_H.grid(row=8, column=2,padx=10,sticky=W)

nty_Ph = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_Ph.grid(row=9, column=2,padx=10,sticky=W)

nty_R = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_R.grid(row=10, column=2,padx=10,sticky=W)

lr = Button(root, text="Predict using RF", command=func_RF,bg="white",fg="Dark red", width=15, padx=80)
lr.config(font=("Times new roman", 16))
lr.grid(row=7, column=3, columnspan=6, padx=10,pady=10, sticky=W)
# lr.place(x="100",y="100")

ref = Button(root, text="Refresh", command=refresh, bg="grey",fg="Dark green", width=15)
ref.config(font=("Times new roman", 16))
ref.grid(row=10, column=4,padx=10,pady=10, sticky=W)

bck = Button(root, text="Go back", command=back, bg="grey",fg="Dark green", width=15)
bck.config(font=("Times new roman", 16))
bck.grid(row=10, column=3, padx=10,pady=10, sticky=W)

root.mainloop()


