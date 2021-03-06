import numpy as np


class Request:

    def __init__(self, auctioned_object, time):
        self.auctioned_object = auctioned_object
        self.time = time


class Auctioneer:
    """
    Implements the auctioneer.

    Attributes:
    ===========
    revenue (float): the sum of payments the auctioneer received in the auction.
    fees_paid (float): the sum of fees the auctioneer paid in the auction.

    Parameters:
    ===========
    auctioned_objects (array of objects): the objects the auctioneer wants to sell in the auction.
    x0 (array): the auctioneers valuation. Default None (valuation is 0 for all objects)
    """

    def __init__(self, auctioned_objects, x0=None):
        if x0 is None:
            self.x0 = np.zeros(auctioned_objects.size)
        else:
            self.x0 = x0
        self.auctioned_objects = auctioned_objects
        self.n_objects = len(auctioned_objects)
        assert self.x0.size == self.auctioned_objects.size, "x0 must be of same size like auctioned_objects"
        self.revenue = 0
        self.fees_paid = 0

    def select_object_to_sell(self):
        """
        Select an object to be sold in the given round of the auction. The probability of being selected is
        proportional to the quality of the object.

        Returns:
        ========
        auctioned_object (AuctionedObject): the instance of the object selected for being auctioned in the given round.

        """
        probabilities = np.array([self.auctioned_objects[i].quality for i in range(self.n_objects)])
        # available quantity check
        quantities = np.array([self.auctioned_objects[i].quantity for i in range(self.n_objects)])
        assert quantities.any() > 0, "No more objects to sell"
        probabilities = np.where(quantities == 0, 0, probabilities)

        probabilities = probabilities / probabilities.sum()
        obj_id = np.random.choice(np.arange(self.n_objects), p=probabilities)
        return self.auctioned_objects[obj_id]

    def send_request(self):
        auctioned_object = self.select_object_to_sell()
        time = 0
        request = Request(auctioned_object, time)
        return request


class AuctionedObject:
    """
    Implements the auctioned object.

    Parameters:
    ===========
    id_ (int): the identification of the object.
    quality (float): describes the value of the object. The higher quality, the better. Objects with higher quality are
                    offered in auctions more frequently.
    quantity (int): the amount of units of the object available for auction.
    fee (float): must be between 0 and 1. The share of the payment the auctioneer must give up if the object is sold.

    """

    def __init__(self, id_, quality=1, quantity=1, fee=0):
        self.id_ = id_
        self.quality = quality
        self.quantity = quantity
        assert (fee >= 0) & (fee <= 1), "The fee must be between 0 and 1."
        self.fee = fee
