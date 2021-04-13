import numpy as np
from scipy import stats, optimize


class Basic:

    def __init__(self):
        self.rp = 0

    def modify_auction(self, auction):
        self.rp = auction.auctioned_object.x0
        if self.rp > auction.winning_bid:
            auction.revenue = auction.auctioned_object.x0
            auction.fee_paid = 0
            auction.sold = False
        return auction


class Myerson:

    def __init__(self, batch_size, batch_sample_size, minprice=0):
        self.batch_size = batch_size
        self.batch_sample_size = batch_sample_size
        self.minprice = minprice

        self.rp = 0
        self.gradient_update_counter = 0
        self.batch_counter = 0

        self._make_data_containers()

    def _make_data_containers(self):
        self.bids_data = np.array([])

    @staticmethod
    def _myerson_formula(r, dist, x0):
        return r - (1 - dist.cdf(r)) / dist.pdf(r) - x0

    def update_step(self, bids):
        self.bids_data = np.concatenate([self.bids_data, bids])
        self.gradient_update_counter += 1

        if self.gradient_update_counter == self.batch_size:
            s = np.random.choice(np.arange(self.bids_data.size), self.batch_sample_size)
            bids_smpl = self.bids_data[s]

            lbids = np.log(bids_smpl)
            mean, std = lbids.mean(), lbids.std()
            dist = stats.lognorm(s=std, loc=1e-5, scale=np.exp(mean))
            x0 = self.minprice
            self.rp = optimize.fsolve(lambda r: self._myerson_formula(r, dist, x0), x0=np.log(bids_smpl).mean())[0]

            self._make_data_containers()
            self.gradient_update_counter = 0

    def predict(self):
        return self.rp

    def modify_auction(self, auction):
        bids = auction.bids
        self.update_step(bids)
        self.rp = self.predict()

        if self.rp > auction.winning_bid:
            auction.revenue = auction.auctioned_object.x0
            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
        return auction


class Appnexus:

    def __init__(self, alpha, eta, n_features=0, beta_init=None, batch_size=1, batch_sample_size=1, burnin_size=1):
        self.beta = beta_init
        self.batch_size = batch_size
        self.batch_sample_size = batch_sample_size
        self.alpha = alpha
        self.eta = eta
        self.n_features = n_features + 1
        self.burnin_size = burnin_size  # number of BATCHES for initial learning

        self.rp = 0
        self.gradient_update_counter = 0
        self.batch_counter = 0

        self.lower = 0
        self.upper = 1

        self._make_data_containers()

    def _make_data_containers(self):
        self.X = np.zeros((self.batch_size, self.n_features))
        self.B1 = np.zeros((self.batch_size, 1))
        self.B2 = np.zeros((self.batch_size, 1))
        self.X0s = np.zeros((self.batch_size, 1))

    def _save_data(self, feature_vector, b1, b2, x0):
        self.X[self.gradient_update_counter, ] = feature_vector
        self.B1[self.gradient_update_counter, ] = b1
        self.B2[self.gradient_update_counter, ] = b2
        self.X0s[self.gradient_update_counter, ] = x0

    def _revenue_function(self, r, b1, b2, x0):
        alpha = self.alpha
        elem1 = r+1/alpha*np.log(1+np.exp(-alpha*(r-b2)))
        elem2 = r-b1+1/alpha*np.log(1+np.exp(-alpha*(r-b1)))
        elem3 = (b1-x0)/(1+np.exp(-alpha*(r-b1)))
        return elem1-elem2-elem3

    def _revenue_function_gradient(self, X, b1, b2, x0):
        beta, alpha = self.beta, self.alpha

        r = X@beta
        elem1 = -np.exp(-alpha * (r - b2)) / (1 + np.exp(-alpha * (r - b2)))
        elem2 = np.exp(-alpha * (r - b1)) / (1 + np.exp(-alpha * (r - b1)))
        elem3 = -alpha * (b1 - x0) * np.exp(-alpha * (r - b1)) / (1 + np.exp(-alpha * (r - b1))) ** 2
        derivative = (elem1 + elem2 + elem3).reshape((self.batch_sample_size, 1))
        return (X*derivative).T.sum(axis=1).reshape(-1, 1)

    def engineer_features(self, auction, b1, b2, x0):
        return np.array([1., auction.auctioned_object.quality])

    @staticmethod
    def _minmax_transform(val, upper, lower):
        return (val-lower)/(upper-lower)

    @staticmethod
    def _reverse_minmax_transform(val, upper, lower):
        return val*(upper-lower)+lower

    def _update_weights(self, X, b1, b2, x0):

        self.upper = np.max(b1)
        self.lower = np.min(b2)

        if (self.batch_counter == self.burnin_size) and (self.gradient_update_counter == self.batch_size):
            b2 = self._minmax_transform(b2, self.upper, self.lower)
            self.beta = np.linalg.lstsq(X, b2, rcond=None)[0]
        elif self.batch_counter > self.burnin_size:
            s = np.random.choice(np.arange(X.shape[0]), self.batch_sample_size)
            X_smpl = X[s, :]
            b1_smpl = self._minmax_transform(b1[s], self.upper, self.lower)
            b2_smpl = self._minmax_transform(b2[s], self.upper, self.lower)
            x0_smpl = x0[s]
            grad = self._revenue_function_gradient(X_smpl, b1_smpl, b2_smpl, x0_smpl)
            self.beta = self.beta + self.eta * grad

    def update_step(self, x, b1, b2, x0):
        # gradient update step
        if self.gradient_update_counter < self.batch_size:
            self._save_data(x, b1, b2, x0)
            self.gradient_update_counter += 1
        else:
            self._update_weights(self.X, self.B1, self.B2, self.X0s)
            self._make_data_containers()

            self.gradient_update_counter = 0
            self.batch_counter += 1

    def predict(self, feature_vector):
        if self.batch_counter <= self.burnin_size:
            return 0
        else:
            rp = feature_vector.reshape((self.n_features, 1)).T@self.beta
            rp = self._reverse_minmax_transform(rp, self.upper, self.lower)
            return rp

    def modify_auction(self, auction):
        b1, b2, x0 = auction.winning_bid, auction.second_bid, auction.auctioned_object.x0

        x = self.engineer_features(auction, b1, b2, x0)

        # update step
        self.update_step(x, b1, b2, x0)

        # prediction step
        self.rp = self.predict(x)

        # modify auction
        if self.rp > auction.winning_bid:
            auction.revenue = auction.auctioned_object.x0
            auction.payment = 0
            auction.fee_paid = 0
            auction.sold = False
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