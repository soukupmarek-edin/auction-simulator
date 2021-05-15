import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder, BreakTakingBidder, LinearBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 20000
n_objects = 1
n_bidders = 20

# auctioned object features
f1 = np.random.uniform(0.1, 1., size=n_objects)
f2 = np.random.binomial(1, p=0.5, size=n_objects)
features = [f1, f2]
d_features = len(features)

auctioned_objects = np.array([AuctionedObject(id_=i, features=np.array([f1[i], f2[i]]), quantity=np.inf) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, selection_rule='random')
bidders = np.array([SimpleBidder(sigma=0.25) for i in range(n_bidders)])
# bidders = np.array([BreakTakingBidder(sigma=0.25) for i in range(n_bidders)])
# bidders = np.array([LinearBidder(d_features, weights=np.random.uniform(-0.1, 0.1, size=d_features), bias=1, sigma=0.5) for _ in range(n_bidders)])

batch_size = 64
sample_size = 64

weights_init = np.zeros(2)
alpha = 250
eta = 0.0006
ufp_target = 0.2
rp_policy = Rp.Appnexus(n_rounds//batch_size, weights_init, batch_size, sample_size,
                        ufp_target=ufp_target, alpha=alpha, eta=eta, x0=0.01,
                        track_hyperparameters=True)

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_policy=rp_policy, track_auctions=True, track_bidders=True)

if __name__ == '__main__':

    house.run()

    bdf = house.bidder_tracker.make_dataframe(variables=['bids'])
    adf = house.auction_tracker.make_dataframe()
    hdf = house.reserve_price_policy.hyperparam_tracker.make_dataframe()

    df = adf.iloc[int(n_rounds/2):]
    spa_rev_share = round(df.payment.sum() / df.second_bid.sum(), 2)
    ufp_share = round(1-df.sold.mean(), 2)
    print("SPA revenue share: ", spa_rev_share)
    print("UFP share: ", ufp_share)

    plt.rc('figure', figsize=(6, 2))
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()

    # adf[['winning_bid', 'second_bid']].plot(ax=ax, alpha=0.4)
    # adf['reserve_price'].plot(c='k', ax=ax, label='reserve price', lw=3)
    # ax.legend()

    adf = adf.assign(batch_id=adf.index // batch_size)
    adf.groupby("batch_id")[['sold']].apply(lambda x: (1 - x.sold).mean()).rolling(20).mean().plot(title='ufp', ax=axes[0])
    axes[0].axhline(ufp_target, c='k', ls='dashed')
    axes[0].set_ylim((0,1))

    hdf.x0.plot(title='x0', ax=axes[1])

    hdf.ufp_tracker.plot(title='ufp tracker', ax=axes[2])

    # plt.tight_layout()
    plt.show()


