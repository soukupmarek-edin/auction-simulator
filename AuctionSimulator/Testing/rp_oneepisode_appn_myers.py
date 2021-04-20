import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 10000
n_objects = 10
n_bidders = 10

# budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])*80
# budgets_p = np.ones(9) / 9
# budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

qualities = np.random.uniform(0.1, 1., n_objects)
# qualities = np.array([0.5, 1.5])
minprices = np.zeros(n_objects)

batch_size = 250
batch_sample_size = 64

alpha = 100
eta = 0.00075

rp_basic = Rp.Basic()

rp_appn = Rp.Appnexus(alpha=alpha, eta=eta,
                        batch_size=batch_size, batch_sample_size=batch_sample_size, n_features=1)

rp_myers = Rp.Myerson(batch_size, batch_sample_size, n_features=1)

if __name__ == '__main__':

    policies = ['Basic', 'Appnexus', 'Myerson']

    plt.rc('figure', figsize=(7, 7))
    fig, axes = plt.subplots(2, 2, sharey=True, sharex=True)
    axes = axes.flatten()

    for i, rp_policy in enumerate([rp_basic, rp_appn, rp_myers]):
        ax = axes[i]

        auctioned_objects = np.array(
            [AuctionedObject(id_=i, quantity=np.inf, bias=qualities[i]) for i in range(n_objects)])
        auctioneer = Auctioneer(auctioned_objects, selection_rule='quality')
        bidders = np.array([SimpleBidder(sigma=0.95) for i in range(n_bidders)])
        house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                        reserve_price_policy=rp_policy, track_auctions=True)
        house.run()

        auc_df = house.auction_tracker.make_dataframe()
        print("SPA revenue ratio: ", np.round(house.auctioneer.revenue / house.spa_revenue, 2))
        print("UFP: ", round(1-house.n_sold/n_rounds, 2))

        auc_df[['winning_bid', 'second_bid']].plot(ax=ax, alpha=0.4)
        auc_df['reserve_price'].plot(c='k', ax=ax, label='reserve price', lw=2.5)
        ax.legend()
        ax.set_title(policies[i])
        # plt.ylim(ymin=0)
    plt.show()

