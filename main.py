import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 1000
n_objects = 100
n_bidders = 130

budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])
budgets_p = np.ones(9) / 9
budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

# fees = np.array([0, 0.3, 0.36, 0.68])
# fees_p = np.array([0.36, 0.14, 0.30, 0.2])
# fees_init = np.array([np.random.choice(fees, p=fees_p) for i in range(n_objects)])

auctioned_objects = np.array([AuctionedObject(i, 2.5, np.inf) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, x0=np.ones(n_objects) * 20)
bidders = np.array([SimpleBidder(budget=budgets_init[i]) for i in range(n_bidders)])

# reserve prices
rp_fun = Rp.gradient_based
rp_params = {'smoothing_rate': 20, 'learning_rate': 0.001}

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                reserve_price_function=rp_fun,
                                reserve_price_function_params=rp_params
                                )

if __name__ == '__main__':

    house.run()

    bid_df = house.bidder_tracker.make_dataframe(variables=['bids', 'budgets'])
    auc_df = house.auction_tracker.make_dataframe()

    plt.rc('figure', figsize=(5, 2))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    time = np.linspace(0, 24, n_rounds)
    hours = auc_df.time.dt.hour
    auc_df[['winning_bid', 'second_bid']].plot(ax=ax1)
    auc_df['reserve_price'].plot(c='k', ax=ax1)
    # bid_df['budgets'].mean(axis=1).plot(ax=ax2)
    ax2.plot(auc_df.time.dt.hour.unique(),
             auc_df.assign(cnt=np.where(auc_df.winner.isna(), 1, 0)).groupby(auc_df.time.dt.hour)['cnt'].sum())
    plt.show()
