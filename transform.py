import numpy as np
import pandas as pd


class CustomTransformer(object):
    """

    """

    def __init__(self):
        super(CustomTransformer, self).__init__()
        self.transform_history = {}

    def fit(self, X):
        # Do Transformations here and save in history
        # Code comes here


        #

        return self

    def transform(self, X):
        # Code Comes here




        #
        X_ = X.to_numpy()

        return X_

    def fit_transform(self, X):
        self.fit(X=X)
        X_transformed = self.transform(X=X)

        return X_transformed

    def inverse_transform(self):
        # Code comes here

        #
        pass
