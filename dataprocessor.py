#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from mpl_toolkits import mplot3d


class DataProcessor:
    
    """
    	Class for reading, preprocessing raw data, and splitting into train and test set.

    	-----------------------------------------------------------
    	Attributes:
    		self.data:		  DataFrame object.  The original raw dataset
            
    		self.X:		  Input vector X (input features)
            
    		self.y:		  Output vector y (classes)

    	-----------------------------------------------------------
    	Functions:
    		__init__:		Initializes DataProcessor object.  Reads, preprocesses raw data

    		datasets:		returns train and test set (20% split).

    """
    def __init__(self):
        #read in dataset
        data = pd.read_csv("data.csv")
        #data cleaning
        y = np.asarray([0 if i==-1 else i for i in data.values[:,0]])
        data['Y']= y
        self.data=data.iloc[:,1:]
        #separate X and y
        self.X= self.data.values[:,0:2]
        self.y= self.data.values[:,2]
        
    
    def datasets(self):
        #split dataset into train and test
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

data= DataProcessor()
X_train, X_test, y_train, y_test= data.datasets()
