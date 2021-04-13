import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 2500
n_objects = 1
n_bidders = 2

# budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])
# budgets_p = np.ones(9) / 9
# budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

qualities = np.random.uniform(1., 1., n_objects)
# qualities = np.array([0.5, 1.5])
x0s = np.zeros(n_objects)
auctioned_objects = np.array([AuctionedObject(id_=i, quality=qualities[i], quantity=np.inf, x0=x0s[i]) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, selection_rule='quality')
bidders = np.array([SimpleBidder(sigma=0.25) for i in range(n_bidders)])

batch_size = 250
batch_sample_size = 50
rp_policy = Rp.Myerson(batch_size, batch_sample_size, minprice=2)

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_policy=rp_policy)

if __name__ == '__main__':

    house.run()

    bid_df = house.bidder_tracker.make_dataframe(variables=['bids', 'budgets'])
    auc_df = house.auction_tracker.make_dataframe()

    plt.rc('figure', figsize=(5, 2))
    fig, ax = plt.subplots()
    auc_df[['winning_bid', 'second_bid']].plot(ax=ax, alpha=0.4)
    auc_df['reserve_price'].plot(c='k', ax=ax)
    # plt.ylim(ymin=0)
    plt.show()

