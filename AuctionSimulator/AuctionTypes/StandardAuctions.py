import numpy as np


class StandardAuction:
    """
    A class defining the rules of standard auctions.
    """

    @staticmethod
    def determine_winner(bids, r=0):
        """
        Selects the winner among bids submitted.
        N - number of bidders

        Parameters:
        ===========
        bids (array): (N,) vector of submitted bids

        Returns:
        ========
        winner_idx (int): the index of the highest bid
        """
        b1 = bids.max()
        if r > b1:
            return None, b1
        else:
            winner_idx = bids.argmax()
            return winner_idx, b1


class SecondPriceAuction(StandardAuction):

    @staticmethod
    def determine_payment(bids, r=0):
        b1 = bids.max()
        b2 = bids[np.argsort(bids)[-2]]
        if b2 >= r:
            return b2, b2
        elif b1 < r:
            return 0, b2
        else:
            return r, b2


class FirstPriceAuction(StandardAuction):

    @staticmethod
    def determine_payment(bids, r=0):
        b1 = bids.max()
        b2 = bids[np.argsort(bids)[-2]]
        if b1 < r:
            return 0, b2
        else:
            return b1, b2
