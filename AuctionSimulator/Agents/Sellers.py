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

    def __init__(self, auctioned_objects, selection_rule='random'):
        self.minprices = np.array([ao.minprice for ao in auctioned_objects])
        self.auctioned_objects = auctioned_objects
        self.selection_rule = selection_rule
        self.n_objects = len(auctioned_objects)
        assert self.minprices.size == self.auctioned_objects.size, "minprices must be of same size like auctioned_objects"
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
        if self.selection_rule == 'random':
            obj_ids = np.array([ao.id_ for ao in self.auctioned_objects])
            quantities = np.array([ao.quantity for ao in self.auctioned_objects])
            selected_id = np.random.choice(obj_ids[quantities > 0])
            return self.auctioned_objects[selected_id]

        elif self.selection_rule == 'quality':
            probabilities = np.array([self.auctioned_objects[i].quality for i in range(self.n_objects)])
            # available quantity check
            quantities = np.array([self.auctioned_objects[i].quantity for i in range(self.n_objects)])
            assert quantities.any() > 0, "No more objects to sell"
            probabilities = np.where(quantities == 0, 0, probabilities)

            probabilities = probabilities / probabilities.sum()
            obj_id = np.random.choice(np.arange(self.n_objects), p=probabilities)
            return self.auctioned_objects[obj_id]
        else:
            raise AttributeError("Unknown selection rule")

    def send_request(self):
        auctioned_object = self.select_object_to_sell()
        time = 0
        request = Request(auctioned_object, time)
        return request


class AuctionedObject:

    def __init__(self, id_, features, weights=None, quantity=1, fee=0, minprice=0):
        self.id_ = id_
        self.quantity = quantity
        assert (fee >= 0) & (fee <= 1), "The fee must be between 0 and 1."
        self.fee = fee
        self.minprice = minprice

        self.quality = 0
        self.features = features
        self.n_features = features.size
        if weights:
            self.quality = np.dot(features, weights)
        else:
            self.quality = np.dot(features, np.ones(features.size))
