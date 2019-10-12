import numpy as np
import csv


def read_data(path) -> np.array:
    data = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # skip header
        next(csv_reader)
        # get all the rows as a list
        data = list(csv_reader)
        data = np.array(data)
    return data
