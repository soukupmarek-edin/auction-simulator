import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder, BreakTakingBidder, LinearBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse
from itertools import product

n_rounds = 20000
n_objects = 10
n_bidders = 10

# auctioned object features
f1 = np.random.uniform(0.1, 1., size=n_objects)
f2 = np.random.binomial(1, p=0.5, size=n_objects)
features = [f1, f2]
d_features = len(features)

auctioned_objects = np.array([AuctionedObject(id_=i, features=np.array([f1[i], f2[i]]), quantity=np.inf) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, selection_rule='random')
# bidders = np.array([BreakTakingBidder(sigma=0.25) for i in range(n_bidders)])
bidders = np.array([LinearBidder(d_features, weights=np.random.uniform(0.8, 1.2, size=d_features), bias=1) for _ in range(n_bidders)])

batch_size = 500
sample_size = 500

categories = np.unique(np.array(list(product(f1, f2))), axis=0)
target_ufps = np.repeat(0.2, categories.shape[0])

rp_policy = Rp.Basic(categories, target_ufps, batch_size, sample_size)

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_policy=rp_policy, track_auctions=True, track_objects=True)

if __name__ == '__main__':

    house.run()

    auc_df = house.auction_tracker.make_dataframe()

    df = auc_df.iloc[int(n_rounds/2):]
    df = df.assign(batch_id=df.index // batch_size)

    spa_rev_share = round(df.payment.sum() / df.second_bid.sum(), 2)
    ufp_share = round(1-df.sold.mean(), 2)
    print("SPA revenue share: ", spa_rev_share)
    print("UFP share: ", ufp_share)

    fig, ax = plt.subplots(figsize=(5,2))
    df[['winning_bid', 'second_bid']].plot(ax=ax, alpha=0.4)
    df['reserve_price'].plot(ax=ax, c='k', lw=2.5)
    plt.show()

