import numpy as np
import pandas as pd
import csv


def read_data(path) -> np.array:
    data = []
    df = pd.read_csv(path, quotechar='"', delimiter=',', skipinitialspace=True)
    data = df.to_numpy()
    # with open(path, 'r') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     # skip header
    #     next(csv_reader)
    #     # get all the rows as a list
    #     data = list(csv_reader)
    #     data = np.array(data)
    return data
