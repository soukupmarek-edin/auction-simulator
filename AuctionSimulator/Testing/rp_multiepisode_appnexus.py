import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.Agents.Sellers import AuctionedObject, Auctioneer
from AuctionSimulator.Agents.Bidders import SimpleBidder
from AuctionSimulator.AuctionHouse import ReservePrice as Rp
from AuctionSimulator.AuctionHouse import AuctionHouse

ENV_CONFIG = {
    'episode_length': 5000
}

APPNEXUS_CONFIG = {
    'batch_size': 250,
    'batch_sample_size': 64,
    'alpha': 10,
    'eta': 0.001,
    'n_features': 1,
    'burnin_size': 1
}


def make_environment(episode_config, policy_config):
    n_rounds = episode_config['episode_length']
    n_objects = 2
    n_bidders = 2

    # budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])
    # budgets_p = np.ones(9) / 9
    # budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

    # fees = np.array([0, 0.3, 0.36, 0.68])
    # fees_p = np.array([0.36, 0.14, 0.30, 0.2])
    # fees_init = np.array([np.random.choice(fees, p=fees_p) for i in range(n_objects)])

    qualities = np.random.uniform(0.5, 1., n_objects)
    # qualities = np.array([0.5, 1.5])
    x0s = np.zeros(n_objects)
    auctioned_objects = np.array([AuctionedObject(id_=i, quality=qualities[i], quantity=np.inf, x0=x0s[i]) for i in range(n_objects)])
    auctioneer = Auctioneer(auctioned_objects, selection_rule='quality')
    bidders = np.array([SimpleBidder(sigma=2) for i in range(n_bidders)])

    rp_policy = Rp.Appnexus(**policy_config)

    house = AuctionHouse.Controller(n_rounds, auctioneer, bidders,
                                    reserve_price_policy=rp_policy)
    return house


def play_episode(episode_counter):

    house = make_environment(ENV_CONFIG, APPNEXUS_CONFIG)
    house.run()

    percent_ufp = (ENV_CONFIG['episode_length'] - house.n_sold)/ENV_CONFIG['episode_length']
    percent_revenue = house.auctioneer.revenue/house.spa_revenue

    return percent_ufp, percent_revenue


if __name__ == '__main__':
    num_episodes = 10
    for ep in range(num_episodes):
        percent_ufp, percent_revenue = play_episode(ep)

        print(f"episode {ep} \n" + "="*20)
        print("ufp: ", percent_ufp)
        print("percent SPA revenue: ", percent_revenue)
        print("="*20)
