import numpy as np
import pandas as pd
from itertools import product


class BidderTracker:
    """
    An object storing bidder-specific data for every auction. Currently stores budgets and bids. If any throttling
    policy is present in the auction, also stores the probabilities of participation and the decision about
    participation.

    Parameters:
    ===========
    n_rounds (int): the number of auctions
    n_bidders (int): the number of bidders
    """

    def __init__(self, n_rounds, n_bidders):
        self.n_rounds = n_rounds
        self.n_bidders = n_bidders
        self.budgets_data = np.zeros((n_rounds, n_bidders))
        self.bids_data = np.zeros((n_rounds, n_bidders))
        self.probabilities_data = np.zeros((n_rounds, n_bidders))
        self.decisions_data = np.zeros((n_rounds, n_bidders))

    def make_dataframe(self, variables=None):
        """
        Creates a data frame with stored bidder-specific data.

        Parameters:
        ===========
        variables (list): the variables to be stored.

        Returns:
        ========
        df (DataFrame): pandas data frame with the stored data.

        """
        if variables is None:
            variables = ['bids', 'budgets', 'probabilities', 'decisions']
        cols = pd.Index(product(variables, np.arange(self.n_bidders)))
        df = pd.DataFrame(columns=cols, index=list(np.arange(self.n_rounds)))

        for variable in variables:
            df.loc[:, variable] = self.__dict__[f'{variable}_data']

        df.index.name = 'auction_round'
        return df


class AuctionTracker:
    """
    An object storing data from the auction. The variables are auction-specific, not bidder-specific. If the Tracker is
    used, the following variables are saved for each auction: sold object id, winner, winning bid, second-highest bid,
    payment, reserve price, fee

    Attributes:
    ===========
    data (array): An array with stored data

    Parameters:
    ===========
    n_rounds (int): the number of auctions that will take place
    """

    def __init__(self, n_rounds):
        self.columns = ['object_id', 'winner', 'winning_bid', 'second_bid', 'payment', 'reserve_price', 'fee']
        self.data = np.zeros((n_rounds, len(self.columns)))
        self.n_rounds = n_rounds
        self.time = self._make_time()

    def _make_time(self):
        dates = pd.date_range('2020-01-01', '2020-01-02', freq='S')
        morning_size = dates[dates.hour <= 5].size
        probabilities = np.concatenate([np.repeat(0.05, morning_size) / morning_size,
                                        np.repeat(0.95, dates.size - morning_size) / (dates.size - morning_size)])
        np.random.seed(42)
        times = np.sort(np.random.choice(dates, p=probabilities, size=self.n_rounds))
        return times

    def make_dataframe(self):
        """
        Returns:
        ========
        df (DataFrame): a pandas data frame with the stored data

        """
        df = pd.DataFrame(self.data, columns=self.columns)

        df['time'] = self.time
        df.index.name = 'auction_round'
        return df
