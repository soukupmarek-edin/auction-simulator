import numpy as np


class Planning:
    """
    Parameters:
    ===========
    n_rounds (int): the number of auctions.
    budgets_init (array): an array of initial budgets of all bidders.
    """

    def __init__(self, n_rounds, budgets_init):
        self.n_rounds = n_rounds
        self.budgets_init = budgets_init
        self.n_bidders = len(budgets_init)

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


class Probability:

    def __init__(self):
        pass

    @staticmethod
    def linear_probability(current_plan, current_budgets, fee, floor=0):
        """
        If the bidder's budget is below their plan, the probability of participation decreases linearly with
        fee up to a specified floor. The probability of participation is always 1 if the budget is above the plan.

        Math: P = 1 - phi * (1-floor)

        Parameters:
        ===========
        current_plan (array): The budget spending plan for the given round.
        floor (float): The lowest probability of participation. I.o.w., the probability of participation
                    if the budget is being spent faster than planned and the fee is 100%.

        Returns:
        ========
        probabilities (array): probabilities of participation for all bidders.
        """
        n_bidders = len(current_budgets)
        assert (floor >= 0) & (floor <= 0), 'The floor must be between 0 and 1.'
        probabilities = np.repeat(1 - fee * (1 - floor), n_bidders)
        probabilities[current_budgets >= current_plan] = 1
        return probabilities

    @staticmethod
    def total_probability(current_plan, current_budgets):
        """
        The bidder will not participate in the auction if their budget is below the current plan

        Parameters:
        ===========
        current_plan (array):
        current_budgets
        """

        probabilities = np.where(current_budgets >= current_plan, 1, 0)
        return probabilities

    @staticmethod
    def budget_probability(current_budgets):
        """
        The probability of participating in the auction is proportional to the remaining budget relative
        to other bidders. Thus the bidder with the largest budget participates with probability equal to 1.
        """

        def min_max_transform(arr):
            return (arr - arr.min()) / (arr.max() - arr.min())

        probabilities = min_max_transform(current_budgets)
        return probabilities


class Decision:

    @staticmethod
    def binomial_decision(probabilities):
        return np.random.binomial(1, 1 - probabilities).astype(bool)
