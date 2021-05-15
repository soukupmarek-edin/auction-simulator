import numpy as np
import pandas as pd
from scipy import stats, optimize
from collections import defaultdict
from abc import ABC, abstractmethod
from AuctionSimulator.Data.Trackers import HyperparameterTracker


def combined_stdev(meanc: float, means: np.ndarray, stdevs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Calculates a combined standard deviation of multiple samples
    :param meanc: combined mean
    :param means:  means of all samples
    :param stdevs: standard deviations of all samples
    :param counts: number of observations in all samples
    :return (float): combined standard deviation
    """
    return np.sqrt(np.sum(counts*(stdevs**2 + (means - meanc)**2))/np.sum(counts))


def standardize(arr):
    return (arr-arr.mean())/arr.std()


def minmax_transform(arr):
    return (arr-arr.min())/(arr.max()-arr.min())


class ReplayBuffer:

    def __init__(self, d_features, batch_size, sample_size=None):
        self.d_features = d_features
        self.d_outcomes = 7  # winning bid, second bid, mean bids, mean log bids, std bids, std log bids, number of bidders
        self.batch_size = batch_size
        if not sample_size:
            self.sample_size = batch_size
        else:
            self.sample_size = sample_size

        self.counter = 0
        self.make_data_containers()

    def make_data_containers(self):
        self.features = np.empty((self.batch_size, self.d_features))
        self.outcomes = np.empty((self.batch_size, self.d_outcomes))

    def save_data(self, features, outcomes):
        """
        Saves one row of data - characteristics of one auction and its outcomes
        :param features: characteristics of the auction, e.g. attributes of the auctioned object
        :param outcomes: continuous numeric outcomes of the auction
        """

        self.features[self.counter, :] = features
        self.outcomes[self.counter, :] = outcomes

        if self.counter == self.batch_size-1:
            self.counter = 0
        else:
            self.counter += 1

    def sample_from_data(self, reset_containers=True):
        s = np.random.choice(np.arange(self.batch_size), self.sample_size)
        f, o = self.features[s, :], self.outcomes[s, :]
        if reset_containers:
            self.make_data_containers()
        return f, o

    def record_auction(self, auction):
        """
        Online learning-specific and environment-specific
        """
        features = auction.auctioned_object.features
        outcomes = [auction.winning_bid,
                    auction.second_bid,
                    auction.bids.mean(),
                    np.log(auction.bids).mean(),
                    auction.bids.std(),
                    np.log(auction.bids).std(),
                    auction.bids.size
                    ]
        self.save_data(features, outcomes)


class ReservePricePolicy(ABC):

    def __init__(self, d_features, batch_size, sample_size):
        self.replay_buffer = ReplayBuffer(d_features, batch_size, sample_size)
        self.d_features = d_features
        self.batch_size = batch_size

        self.counter = 0

    @abstractmethod
    def predict(self, features):
        ...

    @abstractmethod
    def learn(self, feature_batch, outcome_batch):
        ...

    @abstractmethod
    def modify_auction(self, auction):
        ...


class Basic(ReservePricePolicy):

    def __init__(self, categories, target_ufps, batch_size, sample_size):
        super().__init__(categories.shape[1], batch_size, sample_size)
        assert categories.shape[0] == target_ufps.size, "The number of categories must equal the number of target ufps"
        self.categories = categories
        self.n_categories = categories.shape[0]
        self.target_ufps = target_ufps

        self.rp_table = defaultdict(lambda: 0)

    def learn(self, feature_batch, outcome_batch):

        for i in range(self.n_categories):
            category = self.categories[i, :]
            selection = np.all(feature_batch == category, axis=1)
            ufp_target = self.target_ufps[i]
            if np.any(selection):
                bids = outcome_batch[selection, 0]  # select the winning bids for auctions in the given category
                self.rp_table[tuple(category)] = np.quantile(bids, q=ufp_target)

    def predict(self, category):
        return self.rp_table[tuple(category)]

    def modify_auction(self, auction):
        category = auction.auctioned_object.features

        # features and data preparation
        outcome = np.concatenate([np.array([auction.winning_bid]), np.repeat(np.nan, 6)])
        self.replay_buffer.save_data(category, outcome)

        # update step
        if self.counter == self.batch_size-1:
            feature_batch, outcome_batch = self.replay_buffer.sample_from_data()
            self.learn(feature_batch, outcome_batch)
            self.counter = 0
        else:
            self.counter += 1

        # prediction step
        rp = self.predict(category)
        auction.reserve_price = rp

        # auction info update step
        if rp > auction.winning_bid:
            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
        if (auction.auction_type == 'second_price') and (rp < auction.winning_bid) and (rp > auction.second_bid):
            auction.payment = rp
        return auction


class Myerson(ReservePricePolicy):

    def __init__(self, categories, target_ufps, batch_size, sample_size, x0_lr):
        super().__init__(categories.shape[1], batch_size, sample_size)
        assert categories.shape[0] == target_ufps.size, "The number of categories must equal the number of target ufps"
        self.categories = categories
        self.target_ufps = target_ufps
        self.batch_size = batch_size
        self.x0_lr = x0_lr
        self.n_categories = categories.shape[0]

        self.rp_table = defaultdict(lambda: 0)
        self.x0s = defaultdict(lambda: 0)
        self.ufps_counter = defaultdict(lambda: 0)
        self.category_counter = defaultdict(lambda: 0)
        self.counter = 0

    def schedule_hyperparameters(self):
        for c in range(self.n_categories):
            category = tuple(self.categories[c, :])
            if self.category_counter[category] > 0:
                ufp_target = self.target_ufps[c]
                ufp_real = self.ufps_counter[category]/self.category_counter[category]
                if ufp_real < (ufp_target*0.9):
                    self.x0s[category] = self.x0s[category] + (1-ufp_real/ufp_target)*self.x0_lr

        self.ufps_counter = defaultdict(lambda: 0)
        self.category_counter = defaultdict(lambda: 0)

    @staticmethod
    def _myerson_formula(r, dist, x0):
        return r - (1 - dist.cdf(r)) / dist.pdf(r) - x0

    def learn(self, feature_batch, outcome_batch):

        # for each category, calculate the reserve price
        for c in range(self.n_categories):
            category = tuple(self.categories[c, :])
            selection = np.all(feature_batch == category, axis=1)

            if np.any(selection):
                x0 = self.x0s[category]
                means = outcome_batch[selection, 3]
                stdevs = outcome_batch[selection, 5]
                cnts = outcome_batch[selection, 6]

                mean_comb = np.average(means, weights=cnts)
                std_comb = combined_stdev(mean_comb, means, stdevs, cnts)

                dist = stats.lognorm(s=std_comb, loc=1e-5, scale=np.exp(mean_comb))
                rp = optimize.fsolve(lambda r: self._myerson_formula(r, dist, x0), x0=mean_comb)[0]
                self.rp_table[category] = rp

    def predict(self, category):
        return self.rp_table[category]

    def modify_auction(self, auction):

        # features and data preparation
        category = tuple(auction.auctioned_object.features)
        self.category_counter[category] += 1
        # winning bid, second bid, mean bids, mean log bids, std bids, std log bids, number of bidders
        outcomes = np.array([np.nan, np.nan, np.mean(auction.bids), np.log(auction.bids).mean(), np.std(auction.bids), np.log(auction.bids).std(), auction.bids.size])
        self.replay_buffer.save_data(category, outcomes)

        # update step
        if self.counter == self.batch_size-1:
            self.schedule_hyperparameters()
            feature_batch, outcome_batch = self.replay_buffer.sample_from_data()
            self.learn(feature_batch, outcome_batch)
            self.counter = 0
        else:
            self.counter += 1

        # prediction step
        rp = self.predict(category)
        auction.reserve_price = rp

        # auction info update step
        if rp > auction.winning_bid:
            self.ufps_counter[category] += 1

            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
        if (auction.auction_type == 'second_price') and (rp < auction.winning_bid) and (rp > auction.second_bid):
            auction.payment = rp
        return auction


class Appnexus(ReservePricePolicy):

    def __init__(self, n_rounds, weights_init, batch_size, sample_size, ufp_target,
                 alpha=20, eta=0.0001, mu=0, x0=0,
                 track_hyperparameters=True):
        super().__init__(weights_init.size, batch_size, sample_size)
        self.weights = weights_init
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.ufp_target = ufp_target
        self.alpha = alpha
        self.eta = eta
        self.mu = mu
        self.x0 = x0
        self.d_features = weights_init.size

        self.track_hyperparameters = track_hyperparameters
        if track_hyperparameters:
            self.hyperparam_tracker = HyperparameterTracker(n_rounds, 4, ['alpha', 'eta', 'x0', 'ufp_tracker'])

        self.ufp_tracker = 0
        self.ufp_counter = 0
        self.lower, self.upper = 0, 10
        self.counter = 0
        self.batch_counter = 0

    def _revenue_function(self, r, b1, b2, x0=0):
        alpha = self.alpha
        elem1 = r+1/alpha*np.log(1+np.exp(-alpha*(r-b2)))
        elem2 = r-b1+1/alpha*np.log(1+np.exp(-alpha*(r-b1)))
        elem3 = (b1-x0)/(1+np.exp(-alpha*(r-b1)))
        return elem1-elem2-elem3

    def _revenue_function_gradient(self, X, b1, b2, x0=0):
        w, alpha = self.weights, self.alpha
        w = w.reshape((self.d_features, 1))
        r = X@w
        elem1 = -np.exp(-alpha * (r - b2)) / (1 + np.exp(-alpha * (r - b2)))
        elem2 = np.exp(-alpha * (r - b1)) / (1 + np.exp(-alpha * (r - b1)))
        elem3 = -alpha * (b1 - x0) * np.exp(-alpha * (r - b1)) / (1 + np.exp(-alpha * (r - b1))) ** 2
        derivative = (elem1 + elem2 + elem3)
        grad = (X*derivative).sum(axis=0)
        return grad

    def _minmax_transform(self, val):
        upper, lower = self.upper, self.lower
        return (val-lower)/(upper-lower)

    def _reverse_minmax_transform(self, val):
        upper, lower = self.upper, self.lower
        return val*(upper-lower)+lower

    def learn(self, features_batch, outcomes_batch):
        b1, b2 = outcomes_batch[:, [0]], outcomes_batch[:, [1]]
        self.upper = self.upper + 1/(self.batch_counter+1)*(np.max(b1)-self.upper)
        self.lower = self.lower + 1/(self.batch_counter+1)*(np.min(b2)-self.lower)
        b1, b2 = self._minmax_transform(b1), self._minmax_transform(b2)

        if self.batch_counter == 0:  # weights initialization
            self.weights = np.linalg.lstsq(features_batch, b2, rcond=None)[0]
            self.weights = self.weights.flatten()
        else:
            grad = self._revenue_function_gradient(features_batch, b1, b2, x0=self.x0)
            self.weights = self.weights + self.eta * (grad - 2*self.mu*self.weights)

    def predict(self, features: np.ndarray) -> float:
        rp = np.dot(self.weights, features)
        rp = self._reverse_minmax_transform(rp)
        return rp

    def schedule_hyperparameters(self):
        ufp_target = self.ufp_target
        self.ufp_tracker = round(self.ufp_tracker + 0.05*(self.ufp_counter/self.batch_size - self.ufp_tracker), 2)
        self.x0 = self.x0 + (1 - self.ufp_tracker / ufp_target) * 0.0075

        self.ufp_counter = 0

    def modify_auction(self, auction):

        # features and data preparation
        feature_vector = auction.auctioned_object.features
        # winning bid, second bid, mean bids, mean log bids, std bids, std log bids, number of bidders
        outcomes = np.array([auction.winning_bid, auction.second_bid, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.replay_buffer.save_data(feature_vector, outcomes)

        # update step
        if self.counter == self.batch_size - 1:
            # hyperparameter tracker
            if self.track_hyperparameters:
                self.hyperparam_tracker.data[self.batch_counter, :] = (self.alpha, self.eta, self.x0, self.ufp_tracker)
            self.schedule_hyperparameters()
            feature_batch, outcome_batch = self.replay_buffer.sample_from_data()
            self.learn(feature_batch, outcome_batch)
            self.counter = 0
            self.batch_counter += 1
        else:
            self.counter += 1

        # prediction step
        rp = self.predict(feature_vector)
        auction.reserve_price = rp

        # auction info update step
        if rp > auction.winning_bid:
            self.ufp_counter += 1
            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
        if (auction.auction_type == 'second_price') and (rp < auction.winning_bid) and (rp > auction.second_bid):
            auction.payment = rp
        return auction


def rp_fee(realtime_kwargs, **params):
    fee = realtime_kwargs['fee']
    winning_bid = realtime_kwargs['winning_bid']
    x0 = realtime_kwargs['x0']
    r = fee * winning_bid + (1 - fee) * x0
    return r


def one_shot(realtime_kwargs, **params):
    r = realtime_kwargs['current_r']
    b1 = realtime_kwargs['winning_bid']
    b2 = realtime_kwargs['second_bid']
    x0 = realtime_kwargs['x0']

    if r == 0:
        r = np.mean([b1, b2])

    if r > b1:
        return np.max([x0, (1-0.3)*r])
    elif r < b2:
        return np.max([x0, (1+0.02)*r])
    else:
        return np.max([x0, (1+0.01)*r])