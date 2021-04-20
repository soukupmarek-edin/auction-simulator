import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 2500
n_objects = 20
n_bidders = 2

budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])
budgets_p = np.ones(9) / 9
budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

qualities = np.random.uniform(0.5, 1.5, n_objects)
# qualities = np.array([0.5, 1.5])
minprices = np.zeros(n_objects)
auctioned_objects = np.array([AuctionedObject(id_=i, bias=qualities[i], quantity=np.inf) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, selection_rule='quality')
bidders = np.array([SimpleBidder(sigma=0.5) for i in range(n_bidders)])

batch_size = 250
batch_sample_size = 64

alpha = 20
eta = 0.0005
rp_policy = Rp.Appnexus(alpha=alpha, eta=eta,
                        batch_size=batch_size, batch_sample_size=batch_sample_size, n_features=1)

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_policy=rp_policy, track_auctions=True)

if __name__ == '__main__':

    house.run()

    auc_df = house.auction_tracker.make_dataframe()
    print("SPA revenue ratio: ", house.auctioneer.revenue / house.spa_revenue)
    print("UFP: ", 1-house.n_sold/n_rounds)

    plt.rc('figure', figsize=(5, 2))
    fig, ax = plt.subplots()
    auc_df[['winning_bid', 'second_bid']].plot(ax=ax, alpha=0.4)
    auc_df['reserve_price'].plot(c='k', ax=ax, label='reserve price', lw=2.5)
    # plt.ylim(ymin=0)
    plt.show()

