import numpy as np
from .AuctionHouse import Controller


class Planning(Controller):
    """
    Parameters:
    ===========
    n_rounds (int): the number of auctions.
    budgets_init (array): an array of initial budgets of all bidders.
    """

    def __init__(self, n_rounds, budgets_init):
        self.n_rounds = n_rounds
        self.n_bidders = len(budgets_init)
        self.budgets_init = budgets_init

    def uniform_planning(self):
        """
        The spending is evenly distributed over all auctions.

        Math: b_t = B(1-t/T)
        where b_t is the planned budget at round t, B is the total budget, and T is the total number of auctions.

        Returns:
        ========
        plan (array): (n_rounds x n_bidders) matrix of planned budgets.
        """

        plan = np.tile(self.budgets_init, self.n_rounds).reshape((self.n_rounds, self.n_bidders))
        t = np.arange(self.n_rounds)
        plan = plan * (1 - t / self.n_rounds).reshape((self.n_rounds, 1))
        return plan

    def sigmoid_planning(self, s=1., t0=2.):
        """
        The spending is distributed according to an adjusted sigmoid function. The function can be parametrized.

        Math: b_t = B(1-1/(1+exp(-10s/T*(t-T/t0))))

        Parameters:
        ===========
        s (float): the scaling parameter. The lower s, the flatter the plan. Default 1.
        t0 (float): the shifting parameter. Determines when half of the total budget should be spent.
        Default 2 (half of the budget is spent after one half of all auctions are over).

        Returns:
        ========
        plan (array): (n_rounds x n_bidders) matrix of planned budgets.
        """

        budget = np.tile(self.budgets_init, self.n_rounds).reshape((self.n_rounds, self.n_bidders))
        t = np.arange(self.n_rounds)
        scale = (1 - 1 / (1 + np.exp(-s / self.n_rounds * 10 * (t - self.n_rounds / t0)))).reshape(-1, 1)
        plan = budget * scale
        return plan


class Probability(Controller):

    @staticmethod
    def linear_probability(house, plan, **kwargs):
        """
        If the bidder's budget is below their plan, the probability of participation decreases linearly with
        fee up to a specified floor. The probability of participation is always 1 if the budget is above the plan.

        Math: P = 1 - phi * (1-floor)

        Parameters:
        ===========
        house (object): An AuctionHouse object.
        floor (float): The lowest probability of participation. I.o.w., the probability of participation
                    if the budget is being spent faster than planned and the fee is 100%.

        Returns:
        ========
        probabilities (array): probabilities of participation for all bidders.
        """

        if kwargs:
            floor = kwargs['floor']
        else:
            floor = 0
        assert (floor >= 0) & (floor <= 0), 'The floor must be between 0 and 1.'
        fee = house.auctioned_object.fee
        current_budgets = [b.budget for b in house.bidders]
        probabilities = np.repeat(1 - fee * (1 - floor), house.n_bidders)
        probabilities[current_budgets >= plan[house.counter]] = 1
        return probabilities

    @staticmethod
    def total_probability(house, plan):
        """
        The bidder will not participate in the auction if their budget is below the current plan

        Parameters:
        ===========
        house (object): An AuctionHouse object.
        """

        current_budgets = [b.budget for b in house.bidders]
        probabilities = np.where(current_budgets >= plan[house.counter], 1, 0)
        return probabilities

    @staticmethod
    def budget_probability(house):
        """
        The probability of participating in the auction is proportional to the remaining budget relative
        to other bidders. Thus the bidder with the largest budget participates with probability equal to 1.
        """

        def min_max_transform(arr):
            return (arr - arr.min()) / (arr.max() - arr.min())

        current_budgets = np.array([b.budget for b in house.bidders])
        probabilities = min_max_transform(current_budgets)
        return probabilities


class Decision(Controller):

    @staticmethod
    def binomial_decision(house, probabilities):
        return np.random.binomial(1, 1 - probabilities).astype(bool)
