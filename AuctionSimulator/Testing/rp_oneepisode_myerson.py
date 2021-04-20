import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 20000
n_objects = 20
n_bidders = 10

# budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])
# budgets_p = np.ones(9) / 9
# budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

qualities = np.random.uniform(0.5, 1.5, n_objects)
# qualities = np.array([0.9, 1.1])
auctioned_objects = np.array([AuctionedObject(id_=i, bias=qualities[i], quantity=np.inf) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, selection_rule='random')
bidders = np.array([SimpleBidder(sigma=0.1) for i in range(n_bidders)])

batch_size = 250
batch_sample_size = 250

f1 = np.arange(n_objects)
f2 = np.repeat(0, f1.size)
idx = pd.Index(zip(f1, f2))
idx.names = ['object_id', 'dummy']
ufp_target = 0.4
config_df = pd.DataFrame(np.repeat(ufp_target, n_objects), columns=['ufp_target'], index=idx)
config_df['x0'] = np.ones(n_objects)*2.

rp_policy = Rp.Myerson(batch_size, batch_sample_size, config_df)

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_policy=rp_policy, track_auctions=True)

if __name__ == '__main__':

    house.run()

    auc_df = house.auction_tracker.make_dataframe()

    df = auc_df.iloc[batch_size * 10:]
    df['batch_id'] = df.index // batch_size

    spa_rev_share = np.round(df.payment.sum() / df.second_bid.sum(), 2)
    ufp_share = np.mean(df.winning_bid < df.reserve_price).round(2)
    print("SPA revenue share: ", spa_rev_share)
    print("UFP share: ", ufp_share)

    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    auc_df[['winning_bid', 'second_bid']].plot(ax=ax3, alpha=0.4)
    auc_df['reserve_price'].plot(c='k', ax=ax3, label='reserve price', lw=2.5, title='bids and reserve price')
    ax3.legend()

    auc_df['ufp'] = auc_df.reserve_price > auc_df.winning_bid
    auc_df['batch_id'] = auc_df.index // batch_size
    df = auc_df.groupby(['batch_id', 'object_id'])[['ufp', 'x0']].mean()

    df['ufp'].unstack().plot(ax=ax1, title='share under floor price', grid=True, lw=2.5)
    ax1.axhline(ufp_target, c='k', ls='dashed', label='target ufp share')
    ax1.legend()

    df['x0'].unstack().plot(ax=ax2, title='x0 hyperparameter', grid=True, lw=2.5)

    plt.tight_layout()
    plt.show()

