# Based on reading  the book  "Machine Learning for the Quantified Self: On the Art of Learning from Sensory Data"
import copy
import numpy as np


def normalize_dataset(data_table, columns):
    dt_norm = copy.deepcopy(data_table)
    for col in columns:
        dt_norm[col] = (data_table[col] - data_table[col].mean()) / \
            (data_table[col].max() - data_table[col].min())
    return dt_norm
