import numpy as np


class StandardAuction:
    """
    A class defining the rules of standard auctions.
    """

    @staticmethod
    def allocation_rule(bids, return_maxbid=False):
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
        winner_idx = bids.argmax()
        if return_maxbid:
            return winner_idx, b1
        return winner_idx


class SecondPriceAuction(StandardAuction):

    @staticmethod
    def payment_rule(bids):
        b2 = bids[np.argsort(bids)[-2]]
        return b2


class FirstPriceAuction(StandardAuction):

    @staticmethod
    def payment_rule(bids):
        return bids.max()


class Auction(FirstPriceAuction, SecondPriceAuction):
    def __init__(self, auctioned_object, bids, auction_type, get_second_bid=True):
        self.auctioned_object = auctioned_object
        self.bids = bids
        self.auction_type = auction_type
        self.sold = True

        if auction_type == 'second_price':
            self.determine_winner = SecondPriceAuction.allocation_rule
            self.determine_payment = SecondPriceAuction.payment_rule
        elif auction_type == 'first_price':
            self.determine_winner = FirstPriceAuction.allocation_rule
            self.determine_payment = FirstPriceAuction.payment_rule
        else:
            raise AttributeError('unknown auction type')

        self.winner, self.winning_bid = self.determine_winner(bids, return_maxbid=True)
        if get_second_bid:
            self.second_bid = SecondPriceAuction.payment_rule(bids)
        self.payment = self.determine_payment(bids)
        self.revenue = self.payment * (1 - self.auctioned_object.fee)
        self.fee_paid = self.payment * self.auctioned_object.fee
