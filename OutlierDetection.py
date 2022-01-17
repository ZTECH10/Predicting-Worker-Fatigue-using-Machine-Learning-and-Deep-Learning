# # Based on  reading the book  "Machine Learning for the Quantified Self: On the Art of Learning from Sensory Data"

import scipy
import math
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import util
import copy

# Class for outlier detection algorithms based on some distribution of the data. They
# all consider only single points per row (i.e. one column).


class DistributionBasedOutlierDetection:

    # Finds outliers in the specified column of datatable and adds a binary column with
    # the same name extended with '_outlier' that expresses the result per data point.
    def chauvenet(self, data_table, col):
        # Taken partly from: https://www.astro.rug.nl/software/kapteyn/

        # Computer the mean and standard deviation.
        mean = data_table[col].mean()
        std = data_table[col].std()
        N = len(data_table.index)
        criterion = 1.0/(2*N)

        # Consider the deviation for the data points.
        deviation = abs(data_table[col] - mean)/std

        # Express the upper and lower bounds.
        low = -deviation/math.sqrt(2)
        high = deviation/math.sqrt(2)
        prob = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(0, len(data_table.index)):
            # Determine the probability of observing the point
            prob.append(
                1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i])))
            # And mark as an outlier when the probability is below our criterion.
            mask.append(prob[i] < criterion)
        data_table[col + '_outlier'] = mask
        return data_table

    # Fits a mixture model towards the data expressed in col and adds a column with the probability
    # of observing the value given the mixture model.
    def mixture_model(self, data_table, col, n):

        print('Applying mixture models')
        # Fit a mixture model to our data.
        data = data_table[data_table[col].notnull()][col]
        g = GaussianMixture(n_components=n, max_iter=100, n_init=1)
        reshaped_data = np.array(data.values.reshape(-1, 1))
        g.fit(reshaped_data)

        # Predict the probabilities
        probs = g.score_samples(reshaped_data)

        # Create the right data frame and concatenate the two.
        data_probs = pd.DataFrame(
            np.power(10, probs), index=data.index, columns=[col+'_mixture'])

        data_table = pd.concat([data_table, data_probs], axis=1)

        return data_table
