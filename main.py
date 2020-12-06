import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import Throttling as Thr
from AuctionSimulator.AuctionHouse import AuctionHouse

n_rounds = 1000
n_objects = 100
n_bidders = 130

budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])
budgets_p = np.ones(9) / 9
budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

fees = np.array([0, 0.3, 0.36, 0.68])
fees_p = np.array([0.36, 0.14, 0.30, 0.2])
fees_init = np.array([np.random.choice(fees, p=fees_p) for i in range(n_objects)])

auctioned_objects = np.array([AuctionedObject(i, 2.5, np.inf, fee=fees_init[i]) for i in range(n_objects)])
auctioneer = Auctioneer(auctioned_objects, x0=np.ones(n_objects) * 0)
bidders = np.array([SimpleBidder(budget=budgets_init[i]) for i in range(n_bidders)])

# throttling
plan = Thr.Planning(n_rounds, budgets_init).sigmoid_planning(s=0.3, t0=15)
def probability_function(p, b, _): return Thr.Probability().total_probability(p, b)
decision_function = Thr.Decision.binomial_decision

house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                throttling=True, plan=plan, probability_function=probability_function,
                                decision_function=decision_function
                                )

if __name__ == '__main__':

    house.run()

    plt.rc('figure', figsize=(5, 3))
    df = house.bidder_tracker.make_dataframe(variables=['bids', 'budgets'])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    time = np.linspace(0, 24, n_rounds)
    df[['bids']].mean(axis=1).rolling(20).mean().plot(ax=ax1)
    df[['budgets']].mean(axis=1).plot(ax=ax2)
    # ax2.plot(time, plan.mean(axis=1))
    plt.show()
