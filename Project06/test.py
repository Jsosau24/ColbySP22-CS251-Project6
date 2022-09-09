import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors
from sklearn.neighbors import NearestNeighbors
import os
import re


def tokenize_words(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())



path = 'data/enron_dev/'
dic = {}
total = 0

stopWords = ['the', 'i', 'to', 'was', 'of', 'in', 'a', 'and', 'you', 'he','that', 'his', 'it', 'subject']

for folder in os.listdir(path):
    if folder != '.DS_Store':
        folder = os.path.join(path, folder)
        total += len(os.listdir(folder)) 
        for filename in os.listdir(folder):
            file = open(os.path.join(folder, filename), 'r')
            string =  (" ".join(file.read().split()))
            words = tokenize_words(string)
            for i in words:
                if i not in stopWords:
                    if i in dic:
                        dic[i] += 1
                    else:
                        dic[i] = 1
                        
print(dic)
            
            






