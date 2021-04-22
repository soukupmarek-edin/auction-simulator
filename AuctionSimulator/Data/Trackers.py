import numpy as np
import pandas as pd
from itertools import product


class HyperparameterTracker:

    def __init__(self, n_rounds, n_hyperparameters, param_names):
        self.param_names = param_names
        self.data = np.zeros((n_rounds, n_hyperparameters))

    def make_dataframe(self):
        df = pd.DataFrame(self.data, columns=self.param_names)
        return df


class BidderTracker:

    def __init__(self, n_rounds, n_bidders):
        self.n_rounds = n_rounds
        self.n_bidders = n_bidders
        self.budgets_data = np.zeros((n_rounds, n_bidders))
        self.bids_data = np.zeros((n_rounds, n_bidders))
        self.probabilities_data = np.zeros((n_rounds, n_bidders))
        self.decisions_data = np.zeros((n_rounds, n_bidders))

    def make_dataframe(self, variables=None):

        if variables is None:
            variables = ['bids', 'budgets']
        cols = pd.Index(product(variables, np.arange(self.n_bidders)))
        df = pd.DataFrame(columns=cols, index=list(np.arange(self.n_rounds)))

        for variable in variables:
            df.loc[:, variable] = self.__dict__[f'{variable}_data']

        df.index.name = 'auction_round'
        return df


class AuctionTracker:

    def __init__(self, n_rounds):
        self.columns = ['object_id', 'sold', 'winner', 'winning_bid', 'second_bid', 'payment', 'reserve_price', 'fee', 'minprice']
        self.data = np.zeros((n_rounds, len(self.columns)))
        self.n_rounds = n_rounds

    def make_dataframe(self):
        """
        Returns:
        ========
        df (DataFrame): a pandas data frame with the stored data

        """
        df = pd.DataFrame(self.data, columns=self.columns)

        df.index.name = 'auction_round'
        return df


class ObjectTracker:

    def __init__(self, n_rounds, n_features, feature_names=[]):
        if not feature_names:
            self.feature_names = [f"f_{i}" for i in range(n_features)]
        else:
            self.feature_names = feature_names
        self.data = np.zeros((n_rounds, 1+n_features))
        self.n_rounds = n_rounds

    def make_dataframe(self):
        df = pd.DataFrame(self.data, columns=["object_id"] + self.feature_names)

        df.index.name = 'auction_round'
        return df

