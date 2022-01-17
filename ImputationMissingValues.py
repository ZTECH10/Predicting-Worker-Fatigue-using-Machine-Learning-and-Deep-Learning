# Based on  reading the book  "Machine Learning for the Quantified Self: On the Art of Learning from Sensory Data"

class ImputationMissingValues:

    # Impute the mean values in case of missing data.
    def impute_mean(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].mean())
        return dataset

    # Impute the median values in case ff missing data.
    def impute_median(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].median())
        return dataset

    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        # method='bfill': Bfill or backward-fill propagates the first observed non-null value backward until another non-null value is met.
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset
