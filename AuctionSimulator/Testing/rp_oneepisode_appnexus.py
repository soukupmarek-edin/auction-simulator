import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 20000
n_objects = 1
n_bidders = 10

f1 = np.random.uniform(0.1, 1., size=n_objects)
f2 = np.random.binomial(1, p=0.5, size=n_objects)

auctioned_objects = np.array([AuctionedObject(id_=i, features=np.array([f1[i], f2[i]]), quantity=np.inf) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, selection_rule='random')
bidders = np.array([SimpleBidder(sigma=0.25) for i in range(n_bidders)])

batch_size = 64
sample_size = 64

weights_init = np.zeros(2)
alpha = 250
eta = 0.00065
rp_policy = Rp.Appnexus(n_rounds//batch_size, weights_init, batch_size, sample_size,
                        ufp_target=0.2, alpha=alpha, eta=eta, x0=0.01,
                        track_hyperparameters=True)

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_policy=rp_policy, track_auctions=True)

if __name__ == '__main__':

    house.run()

    auc_df = house.auction_tracker.make_dataframe()

    df = auc_df.iloc[int(n_rounds/2):]
    # df = auc_df.copy()
    df = df.assign(batch_id=df.index // batch_size)

    hyp_df = house.reserve_price_policy.hyperparam_tracker.make_dataframe()
    print(hyp_df)

    spa_rev_share = round(df.payment.sum() / df.second_bid.sum(), 2)
    ufp_share = round(1-df.sold.mean(), 2)
    print("SPA revenue share: ", spa_rev_share)
    print("UFP share: ", ufp_share)

    fig, ax = plt.subplots(figsize=(5, 2))
    df[['winning_bid', 'second_bid']].plot(ax=ax, alpha=0.4)
    df['reserve_price'].plot(ax=ax, c='k', lw=2.5)
    plt.show()

