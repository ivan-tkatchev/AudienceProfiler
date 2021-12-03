import pandas as pd
import numpy as np
from proccessing_bundle import processing_bundle
from processing_device import processing_device
from eval import predict


#Pycharm printing consts
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)



def get_data():
    for df in pd.read_csv("train.csv", chunksize=1000000):
        data = df
        break
    return data

if __name__ == '__main__':
    data = get_data()

    # proccessing bundle's by Andrew
    data = processing_bundle(data)
    data = processing_device(data)
    predict(data)
    print(data.head())

