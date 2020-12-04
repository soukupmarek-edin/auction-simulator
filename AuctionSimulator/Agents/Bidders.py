import numpy as np


class SimpleBidder:
    """
    Implements a Simple Bidder agent. The agent's bidding strategy is to bid a random value drawn from the log-normal
    distribution. The mean equals the quality of the auctioned object and the variance is 0.5.

    Attributes:
    ==========
    wins (int): the counter of agent's wins.
    objects_bought (array of int): a counter of the amount of objects the bidder has bought in the auction

    Parameters:
    ===========
    budget (float): the budget the agent can spend in the auction. Default inf

    """

    def __init__(self, budget=False):
        self.wins = 0
        self.objects_bought = None
        if not budget:
            self.budget = np.inf
        else:
            self.budget = budget

    def submit_bid(self, auctioned_object):
        """
        Implementation of the agent's bidding strategy.

        Parameters:
        ===========
        auctioned_object (object): must have attribute "quality".

        Returns:
        ========
        bid (float): the minimum of the bid chosen by the bidding strategy and the available budget.

        """
        bid = np.exp(np.random.normal(loc=auctioned_object.quality, scale=0.5))
        bid = min(bid, self.budget)
        return bid


class BidderWithPreferences:
    """
    Implements a bidder-with-preferences agent. The bidder is assigned preferences over the auctioned objects.
    The agent's bidding strategy is to bid a random value drawn from the log-normal distribution.
    The expected value equals the mean of the quality of the auctioned object and the agent's preference for that
    object. The variance is 0.5.

    Attributes:
    ==========
    wins (int): the counter of agent's wins.
    objects_bought (array of int): a counter of the amount of objects the bidder has bought in the auction

    Parameters:
    ===========
    preferences (array): an array of floats or integers describing the agent's preferences. The higher number,
                        the higher preference for that object.
    budget (float): the budget the agent can spend in the auction. Default inf/

    """

    def __init__(self, preferences, budget=False):
        self.wins = 0
        self.objects_bought = None
        if not budget:
            self.budget = np.inf
        else:
            self.budget = budget
        self.preferences = preferences

    def submit_bid(self, auctioned_object):
        """
        Implementation of the agent's bidding strategy.

        Parameters:
        ===========
        auctioned_object (object): must have attribute "quality".

        Returns:
        ========
        bid (float): the minimum of the bid chosen by the bidding strategy and the available budget.

        """
        mid = np.mean([auctioned_object.quality, self.preferences[auctioned_object.id_]])
        bid = np.exp(np.random.normal(loc=mid, scale=0.5))
        return min(bid, self.budget)
