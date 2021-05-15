import numpy as np
from abc import ABC, abstractmethod


class Bidder(ABC):

    def __init__(self, budget=False):
        if not budget:
            self.budget = np.inf
        else:
            self.budget = budget

    @abstractmethod
    def submit_bid(self, auctioned_object):
        ...


class LinearBidder(Bidder):

    def __init__(self, d_features, weights, bias=0, sigma=1., budget=False):
        super().__init__(budget)
        self.weights = weights
        self.bias = bias
        self.sigma = sigma

    def submit_bid(self, auctioned_object):
        feature_vector = auctioned_object.features
        bid = np.exp(np.random.normal(loc=self.bias + np.dot(self.weights, feature_vector), scale=self.sigma))
        return max(0, min(bid, self.budget))


class BreakTakingBidder(Bidder):

    def __init__(self, budget=False, sigma=0.5, break_freq=0.05):
        super().__init__(budget)
        self.sigma = sigma
        self.break_freq = break_freq
        self.state = 1

    def submit_bid(self, auctioned_object):
        if np.random.uniform() < self.break_freq:
            self.state *= -1

        if self.state == 1:
            bid = np.random.lognormal(mean=auctioned_object.quality, sigma=self.sigma)
            bid = min(bid, self.budget)
            return bid
        elif self.state == -1:
            return 0


class SimpleBidder(Bidder):

    def __init__(self, budget=False, sigma=0.5):
        super().__init__(budget)
        self.sigma = sigma

    def submit_bid(self, auctioned_object):
        bid = np.random.lognormal(mean=auctioned_object.quality, sigma=self.sigma)
        bid = min(bid, self.budget)
        return bid


class BidderWithPreferences:

    def __init__(self, preferences, budget=False):
        self.wins = 0
        self.objects_bought = None
        if not budget:
            self.budget = np.inf
        else:
            self.budget = budget
        self.preferences = preferences

    def submit_bid(self, auctioned_object):

        mid = np.mean([auctioned_object.quality, self.preferences[auctioned_object.id_]])
        bid = np.exp(np.random.normal(loc=mid, scale=0.5))
        return min(bid, self.budget)


class GuaranteedCampaign:

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
        self.bids = np.zeros(n_rounds)

    def _make_uniform_plan(self):
        plan = np.linspace(0, self.target_displays, self.n_rounds)
        return plan


class GuaranteedBidderRatio(GuaranteedCampaign):

    def submit_bid(self, auctioned_object):
        slack = self.plan[self.round] - self.wins
        self.slacks[self.round] = slack

        if self.plan[self.round] > 0:
            self.b = self.b - self.b*self.kwargs['learning_rate']*(self.wins/self.plan[self.round]-1)

        self.round += 1
        return self.b


class GuaranteedBidderGradient(GuaranteedCampaign):
    """
    This bidder uses the QuadQuad loss function
    """

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


class GuaranteedAllocation(GuaranteedCampaign):

    def decide_allocation(self, current_payment):
        slack = self.plan[self.round] - self.wins
        self.slacks[self.round] = slack
        self.bids[self.round] = self.b

        if slack >= 0:
            self.b = self.b + self.b*self.kwargs['learning_rate']*self.kwargs['dist'].pdf(current_payment)
        else:
            self.b = self.b - self.b*self.kwargs['learning_rate']*self.kwargs['dist'].pdf(current_payment)

        self.round += 1
        decision = int(self.b > current_payment)
        self.wins += decision
        return decision


class GuaranteedAllocation2(GuaranteedCampaign):

    def decide_allocation(self, current_payment):
        slack = self.plan[self.round] - self.wins
        ratio_left = (self.target_displays - self.wins) / (self.n_rounds - self.round)
        self.slacks[self.round] = ratio_left
        self.bids[self.round] = self.b
        dist = self.kwargs['dist']

        self.b = dist.ppf(ratio_left)
        decision = int(self.b > current_payment)
        self.wins += decision
        self.round += 1
        return decision
