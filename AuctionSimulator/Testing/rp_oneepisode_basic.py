import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse
from itertools import product

n_rounds = 20000
n_objects = 20
n_bidders = 10

qualities = np.random.uniform(0.5, 1.5, n_objects)
# qualities = np.array([0.5, 1.5])
auctioned_objects = np.array([AuctionedObject(id_=i, bias=qualities[i], quantity=np.inf) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, selection_rule='random')
bidders = np.array([SimpleBidder(sigma=0.1) for i in range(n_bidders)])

batch_size = 500
batch_sample_size = 500

f1 = np.arange(n_objects)
f2 = np.repeat(0, f1.size)
categories = np.unique(np.array(list(product(f1, f2))), axis=0)
target_ufps = np.repeat(0.4, n_objects)

rp_policy = Rp.Basic(batch_size, batch_sample_size, categories, target_ufps)

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_policy=rp_policy, track_auctions=True)

if __name__ == '__main__':

    house.run()

    auc_df = house.auction_tracker.make_dataframe()

    df = auc_df.iloc[batch_size * 2:]
    df['batch_id'] = df.index // batch_size

    spa_rev_share = np.round(df.payment.sum() / df.second_bid.sum(), 2)
    ufp_share = np.mean(df.winning_bid < df.reserve_price).round(2)
    print("SPA revenue share: ", spa_rev_share)
    print("UFP share: ", ufp_share)

    fig, ax1 = plt.subplots(figsize=(5, 2))

    auc_df[['winning_bid', 'second_bid']].plot(ax=ax1, alpha=0.4)
    auc_df['reserve_price'].plot(c='k', ax=ax1, label='reserve price', lw=2.5, title='bids and reserve price')
    ax1.legend()

    plt.tight_layout()
    plt.show()

