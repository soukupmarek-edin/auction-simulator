import numpy as np
from AuctionSimulator.AuctionHouse import functions


class Planning:
    """
    Parameters:
    ===========
    n_rounds (int): the number of auctions.
    budgets_init (array): an array of initial budgets of all bidders.
    """

    def __init__(self, n_rounds, budgets_init):
        self.budgets_init = budgets_init
        self.n_bidders = len(budgets_init)
        self.time = np.linspace(0, 24, n_rounds)
        self.n_rounds = n_rounds

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

    def sigmoid_planning(self, s=1., t0=12.):
        """
        The spending is distributed according to an adjusted sigmoid function. The function can be parametrized.

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
        scale = 1 - 1 / (1 + np.exp(-s * (self.time - t0)))
        plan = budget * scale.reshape(-1, 1)
        return plan

    def empirical_planning(self, shift=0.25):
        time = self.time
        p1 = 1 - shift / 24 * time[time <= 6]
        p2 = 1. + shift - (1. + shift) / 24 * time[time > 6]
        scale = np.concatenate([p1, p2])
        budget = np.tile(self.budgets_init, self.n_rounds).reshape((self.n_rounds, self.n_bidders))
        plan = budget * scale.reshape(-1, 1)
        return plan


class Probability:

    def __init__(self, **parameters):
        self.parameters = parameters
        self.gaps = np.array([])

    def linear_probability(self, realtime_data):
        """
        If the bidder's budget is below their plan, the probability of participation decreases linearly with
        fee up to a specified floor. The probability of participation is always 1 if the budget is above the plan.

        Math: P = 1 - phi * (1-floor)

        Parameters:
        ===========
        current_plan (array): The budget spending plan for the given round.
        current_budgets (array): Budget of every bidder in the given round
        fee (float):
        floor (float): The lowest probability of participation. I.o.w., the probability of participation
                    if the budget is being spent faster than planned and the fee is 100%.

        Returns:
        ========
        probabilities (array): probabilities of participation for all bidders.
        """
        current_plan = realtime_data['current_plan']
        current_budgets = realtime_data['current_budgets']
        fee = realtime_data['fee']
        floor = self.parameters['floor']
        n_bidders = len(current_budgets)
        assert (floor >= 0) & (floor <= 1), 'The floor must be between 0 and 1.'

        probabilities = np.repeat(1 - fee * (1 - floor), n_bidders)
        probabilities[current_budgets >= current_plan] = 1
        return probabilities

    def lost_revenue_probability(self, realtime_data):
        """
        Kwargs:
        =======
        bids (array)
        fee (float)

        Parameters:
        ===========
        s (float)
        """
        current_plan = realtime_data['current_plan']
        current_budgets = realtime_data['current_budgets']
        bids = realtime_data['bids']
        fee = realtime_data['fee']
        s = self.parameters['s']

        if fee == 0.:
            return np.ones(bids.size)

        standardized_bids = (bids-bids.mean())/bids.std()
        probabilities = functions.reverse_sigmoid(standardized_bids, s)*(1-fee)
        probabilities = np.where(current_budgets >= current_plan, 1, probabilities)
        return probabilities

    def combined_probability(self, realtime_data):
        current_plan = realtime_data['current_plan']
        current_budgets = realtime_data['current_budgets']
        bids = realtime_data['bids']
        fee = realtime_data['fee']

        fees_center = self.parameters['fees_center']
        gap_weight = self.parameters['gap_weight']
        gap_s = self.parameters['gap_s']
        bid_s = self.parameters['bid_s']

        gap = (current_budgets - current_plan) / current_plan
        p_gap = np.where(gap > 0, 1, np.exp(gap_s*gap))

        bids_hat = functions.standardize(bids)
        score = bids_hat*(fee-fees_center)
        p_fee = functions.reverse_sigmoid(score, bid_s)
        return gap_weight*p_gap + (1-gap_weight)*functions.reverse_sigmoid(p_fee, bid_s)

    @staticmethod
    def total_fee_probability(realtime_data):
        current_plan = realtime_data['current_plan']
        n_bidders = current_plan.size
        fee = realtime_data['fee']
        if fee == 0.:
            return np.ones(n_bidders)
        else:
            return np.zeros(n_bidders)

    def total_plan_probability(self, realtime_data):
        """
        The bidder will participate in the auction with specified probability if their budget is below the current plan

        Parameters:
        ===========
        current_plan (array): The budget spending plan for the given round.
        current_budgets (array): Budget of every bidder in the given round.
        prob_under_plan (float):
        """
        current_plan = realtime_data['current_plan']
        current_budgets = realtime_data['current_budgets']
        prob_under_plan = self.parameters['prob_under_plan']
        probabilities = np.where(current_budgets >= current_plan, 1, prob_under_plan)
        return probabilities

    @staticmethod
    def budget_probability(realtime_data):
        """
        The probability of participating in the auction is proportional to the remaining budget relative
        to other bidders. Thus the bidder with the largest budget participates with probability equal to 1.
        """
        current_budgets = realtime_data['current_budgets']

        probabilities = functions.min_max_transform(current_budgets)
        return probabilities


class Decision:

    @staticmethod
    def binomial_decision(probabilities):
        return np.random.binomial(1, 1 - probabilities).astype(bool)
