from tkinter import *
import os
# from gui_stuff import *


import pandas as pd
import numpy as np
from PIL import ImageTk,Image

class Main:
    def __init__(self, root):
        self.window = root
        self.window.title("Predictlizer")
        self.window.geometry("1280x800+0+0")
     
        self.head = Label(root, text="Predictlizer", fg="Dark green" )
        self.head.config(font=("Elephant", 32,))
        self.head.grid(row=1, column=0, columnspan=4, padx=100)
        self.head.place(x=430,y=45)
        
        self.head2 = Label(root, text="Crop & Fertilizer Predictor", fg="green" )
        self.head2.config(font=("Elephant", 20))
        self.head2.grid(row=2, column=0, columnspan=4, padx=100)
        self.head2.place(x=390,y=190)

        self.btn_cp = Button(root, text="Predict Crop", command=self.run_cp,bg="White",fg="red", pady=30, width=25)
        self.btn_cp.config(font=("Times new roman", 22))
        self.btn_cp.grid(row=4, column=2,padx=50,pady=20)
        self.btn_cp.place(x=380,y=250)
   
        self.btn_fp = Button(root, text="Predict Fertilizer", command=self.run_fp,bg="White",fg="red", pady=30, width=25)
        self.btn_fp.config(font=("Times new roman", 22))
        self.btn_fp.grid(row=4, column=2,padx=50,pady=20)
        self.btn_fp.place(x=380,y=400)  
  
    def run_cp(self):
        self.window.destroy()
        import crop_predictor.py

    def run_fp(self):
        self.window.destroy()
        import fertilizer_predictor.py

if __name__ == "__main__":
    root = Tk()
    obj = Main(root)
    root.mainloop()