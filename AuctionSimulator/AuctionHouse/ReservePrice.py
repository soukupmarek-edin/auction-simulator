import numpy as np
import pandas as pd
from scipy import stats, optimize
from collections import defaultdict


class Basic:

    def __init__(self, batch_size, batch_sample_size, categories, target_ufps):
        self.batch_size = batch_size
        self.batch_sample_size = batch_sample_size
        self.n_categories = categories.shape[0]
        self.categories = categories
        self.target_ufps = target_ufps

        self.rp_table = defaultdict(lambda: 0)
        self.update_counter = 0
        self.batch_counter = 0

        self._make_data_containers()

    def _make_data_containers(self):
        self.highest_bids = np.zeros(self.batch_size)
        self.categories = np.zeros((self.batch_size, self.categories.shape[1]))

    def _save_data(self, winning_bid, category):
        self.highest_bids[self.update_counter] = winning_bid
        self.categories[self.update_counter, :] = category

    def update_step(self, bids, category):
        # data update
        self._save_data(bids, category)
        self.update_counter += 1

        if self.update_counter == self.batch_size:

            for i in range(self.n_categories):
                category = self.categories[i, :]
                selection = np.all(self.categories == category, axis=1)
                ufp_target = self.target_ufps[i]
                bids = self.highest_bids[selection]
                self.rp_table[tuple(category)] = np.quantile(bids, q=ufp_target)

            self._make_data_containers()
            self.update_counter = 0

    def predict(self, category):
        return self.rp_table[category]

    def modify_auction(self, auction):

        # features and data preparation
        winning_bid = auction.winning_bid
        category = (auction.auctioned_object.id_, 0)

        # update step
        self.update_step(winning_bid, category)

        # prediction step
        self.rp = self.predict(category)

        # auction info update step
        if self.rp > auction.winning_bid:

            auction.revenue = 0
            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
        if (auction.auction_type == 'second_price') and (self.rp < auction.winning_bid) and (self.rp > auction.second_bid):
            auction.payment = self.rp
            auction.revenue = self.rp
        return auction


class Myerson:

    def __init__(self, batch_size, batch_sample_size, config_df):
        self.batch_size = batch_size
        self.batch_sample_size = batch_sample_size
        self.n_categories = config_df.index.nlevels
        self.categories_names = list(config_df.index.names)

        self.rp = 0
        self.rp_table = defaultdict(lambda: 0)
        self.config_df = config_df
        self.data_df = config_df.copy()
        self.data_df['auctioned'] = 0
        self.data_df['ufp'] = 0
        self.gradient_update_counter = 0
        self.batch_counter = 0

        self._make_data_containers()

    def _make_data_containers(self):
        self.bids_data = np.array([])
        self.auction_stats = np.zeros((self.batch_size, 3))
        self.categories = np.zeros((self.batch_size, self.n_categories))

    def _save_data(self, bids, category):
        bids = np.log(bids)
        mean, std, count = bids.mean(), bids.std(), bids.size
        self.auction_stats[self.gradient_update_counter, :] = [mean, std, count]
        self.categories[self.gradient_update_counter, :] = category

    def schedule_hyperparameters(self):
        for category in self.data_df.index:
            ufp_target = self.data_df.loc[category, 'ufp_target']
            ufp_real = self.data_df.loc[category, 'ufp'] / self.data_df.loc[category, 'auctioned']
            self.data_df.loc[category, 'x0'] = self.data_df.loc[category, 'x0'] + (1-ufp_real/ufp_target)*0.075

        self.data_df['ufp'] = 0
        self.data_df['auctioned'] = 0

    @staticmethod
    def _myerson_formula(r, dist, x0):
        return r - (1 - dist.cdf(r)) / dist.pdf(r) - x0

    def update_step(self, bids, category):
        # data update
        self._save_data(bids, category)
        self.gradient_update_counter += 1
        self.data_df.loc[category, 'auctioned'] += 1

        if self.gradient_update_counter == self.batch_size:
            # create batch data
            index = pd.MultiIndex.from_arrays(self.categories.T)
            index.names = self.categories_names
            df = pd.DataFrame(self.auction_stats, index=index).sample(n=self.batch_sample_size)

            # calculate means and standard deviations
            # wa = lambda x: np.average(x, weights=df.loc[x.index, self.n_features+3])
            df = df.groupby(self.categories_names).mean().sort_index()
            df['rp'] = 0
            df.columns = ['mean', 'std', 'count', 'rp']

            # for each category, calculate the reserve price
            for i, idx in enumerate(df.index):
                mean, std = df.loc[idx, ['mean', 'std']]
                x0 = self.data_df.loc[idx, 'x0']
                dist = stats.lognorm(s=std, loc=1e-5, scale=np.exp(mean))
                self.rp_table[idx] = optimize.fsolve(lambda r: self._myerson_formula(r, dist, x0), x0=mean)[0]

            self.schedule_hyperparameters()

            self._make_data_containers()
            self.gradient_update_counter = 0

    def predict(self, category):
        return self.rp_table[category]

    def modify_auction(self, auction):

        # features and data preparation
        bids = auction.bids
        category = (auction.auctioned_object.id_, 0)
        x0 = self.data_df.loc[category, 'x0']

        # update step
        self.update_step(bids, category)

        # prediction step
        self.rp = self.predict(category)

        # auction info update step
        if self.rp > auction.winning_bid:
            self.data_df.loc[category, 'ufp'] += 1

            auction.auctioned_object.minprice = x0
            auction.revenue = 0
            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
        if (auction.auction_type == 'second_price') and (self.rp < auction.winning_bid) and (self.rp > auction.second_bid):
            auction.payment = self.rp
            auction.revenue = self.rp
        return auction


class Appnexus:

    def __init__(self, alpha=20, eta=0.0001, mu=0, n_features=0, beta_init=None, batch_size=1, batch_sample_size=1, burnin_size=1):
        self.beta = beta_init
        self.batch_size = batch_size
        self.batch_sample_size = batch_sample_size
        self.alpha = alpha
        self.eta = eta
        self.mu = mu
        self.n_features = n_features + 1
        self.burnin_size = burnin_size  # number of BATCHES for initial learning

        self.rp = 0
        self.gradient_update_counter = 0
        self.batch_counter = 0

        self.lower = 0
        self.upper = 10

        self._make_data_containers()

    def _make_data_containers(self):
        self.X = np.zeros((self.batch_size, self.n_features))
        self.B1 = np.zeros((self.batch_size, 1))
        self.B2 = np.zeros((self.batch_size, 1))

    def _save_data(self, feature_vector, b1, b2):
        self.X[self.gradient_update_counter, ] = feature_vector
        self.B1[self.gradient_update_counter, ] = b1
        self.B2[self.gradient_update_counter, ] = b2

    def _revenue_function(self, r, b1, b2, x0=0):
        alpha = self.alpha
        elem1 = r+1/alpha*np.log(1+np.exp(-alpha*(r-b2)))
        elem2 = r-b1+1/alpha*np.log(1+np.exp(-alpha*(r-b1)))
        elem3 = (b1-x0)/(1+np.exp(-alpha*(r-b1)))
        return elem1-elem2-elem3

    def _revenue_function_gradient(self, X, b1, b2, x0=0):
        beta, alpha = self.beta, self.alpha

        r = X@beta
        elem1 = -np.exp(-alpha * (r - b2)) / (1 + np.exp(-alpha * (r - b2)))
        elem2 = np.exp(-alpha * (r - b1)) / (1 + np.exp(-alpha * (r - b1)))
        elem3 = -alpha * (b1 - x0) * np.exp(-alpha * (r - b1)) / (1 + np.exp(-alpha * (r - b1))) ** 2
        derivative = (elem1 + elem2 + elem3).reshape((self.batch_sample_size, 1))
        return (X*derivative).T.sum(axis=1).reshape(-1, 1)

    def engineer_features(self, auction, b1, b2):
        return np.array([1., auction.auctioned_object.quality])

    @staticmethod
    def _minmax_transform(val, upper, lower):
        return (val-lower)/(upper-lower)

    @staticmethod
    def _reverse_minmax_transform(val, upper, lower):
        return val*(upper-lower)+lower

    def _update_weights(self, X, b1, b2, x0=0.):

        self.upper = self.upper + 1/(self.batch_counter+1)*(np.max(b1)-self.upper)
        self.lower = self.lower + 1/(self.batch_counter+1)*(np.min(b2)-self.lower)

        if (self.batch_counter == self.burnin_size) and (self.gradient_update_counter == 0):
            b2 = self._minmax_transform(b2, self.upper, self.lower)
            self.beta = np.linalg.lstsq(X, b2/2, rcond=None)[0]

        if self.batch_counter >= self.burnin_size:
            s = np.random.choice(np.arange(X.shape[0]), self.batch_sample_size)
            X_smpl = X[s, :]
            b1_smpl = self._minmax_transform(b1[s], self.upper, self.lower)
            b2_smpl = self._minmax_transform(b2[s], self.upper, self.lower)
            grad = self._revenue_function_gradient(X_smpl, b1_smpl, b2_smpl, x0)
            self.beta = self.beta + self.eta * (grad - 2*self.mu*self.beta)

    def update_step(self, x, b1, b2):
        # gradient update step
        if self.gradient_update_counter < self.batch_size:
            self._save_data(x, b1, b2)
            self.gradient_update_counter += 1
        else:
            self.gradient_update_counter = 0
            self.batch_counter += 1

            self._update_weights(self.X, self.B1, self.B2)
            self._make_data_containers()

    def predict(self, feature_vector):
        if self.batch_counter < self.burnin_size:
            return 0
        else:
            rp = feature_vector.reshape((self.n_features, 1)).T@self.beta
            rp = self._reverse_minmax_transform(rp, self.upper, self.lower)
            return rp

    def modify_auction(self, auction):
        b1, b2 = auction.winning_bid, auction.second_bid

        x = self.engineer_features(auction, b1, b2)

        # update step
        self.update_step(x, b1, b2)

        # prediction step
        self.rp = self.predict(x)

        # modify auction
        if self.rp > auction.winning_bid:
            auction.revenue = 0
            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
        if (auction.auction_type == 'second_price') and (self.rp < auction.winning_bid) and (self.rp > auction.second_bid):
            auction.payment = self.rp[0][0]
            auction.revenue = self.rp[0][0]
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