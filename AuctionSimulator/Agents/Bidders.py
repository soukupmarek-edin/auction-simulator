import numpy as np


class GuaranteedBidder:

    def __init__(self, target_displays, n_rounds, plan_type='uniform', b0=1, **kwargs):
        self.budget = np.inf
        self.target_displays = target_displays
        self.n_rounds = n_rounds
        if plan_type == 'uniform':
            self.plan = self._make_uniform_plan()
        else:
            raise AttributeError("unknown plan type")
        self.b = b0
        self.kwargs = kwargs
        self.round = 0
        self.wins = 0
        self.slacks = np.zeros(n_rounds)

    def _make_uniform_plan(self):
        plan = np.linspace(0, self.target_displays, self.n_rounds)
        return plan


class GuaranteedBidderGradient(GuaranteedBidder):

    def submit_bid(self, auctioned_object):
        slack = self.plan[self.round]-self.wins
        self.slacks[self.round] = slack

        def gradient(x, alpha):
            return 4*(alpha+(1-2*alpha)*(x > 0))*x

        if (self.plan[self.round]+self.wins) > 0:
            error = (self.plan[self.round]-self.wins)/(self.plan[self.round]+self.wins)
        else:
            error = 0

        self.b = self.b + self.kwargs['learning_rate']*gradient(error, alpha=self.kwargs['cost_shape'])

        self.round += 1
        return self.b


class GuaranteedBidderRatio(GuaranteedBidder):

    def submit_bid(self, auctioned_object):
        slack = self.plan[self.round] - self.wins
        self.slacks[self.round] = slack

        if self.plan[self.round] > 0:
            self.b = self.b - self.b*self.kwargs['learning_rate']*(self.wins/self.plan[self.round]-1)

        self.round += 1
        return self.b


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

    def __init__(self, budget=False, scale=0.5):
        self.wins = 0
        self.objects_bought = None
        if not budget:
            self.budget = np.inf
        else:
            self.budget = budget
        self.scale = scale

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
        bid = np.exp(np.random.normal(loc=auctioned_object.quality, scale=self.scale))
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
