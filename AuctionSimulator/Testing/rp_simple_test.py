import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.AuctionHouse.ReservePrice import Myerson, Appnexus
import pandas as pd

# budgets = np.array([30., 50., 80., 100., 120., 200., 300., 500., 1000.])
# budgets_p = np.ones(9) / 9
# budgets_init = np.array([np.random.choice(budgets, p=budgets_p) for _ in range(n_bidders)])

n_rounds = 1000
n_bidders = 50
x0 = 0
bids = np.random.lognormal(mean=5, sigma=1, size=(n_rounds, n_bidders))
bids = np.sort(bids, axis=1)[:, ::-1]

rp_policy = Appnexus(alpha=10, eta=0.05, beta_init=np.array([[1.]]))
# rp_policy = Myerson(100, 100)


if __name__ == '__main__':
    predicted_rp = np.zeros(n_rounds)
    for i, a in enumerate(bids):
        b1 = a[0]
        b2 = a[1]

        # gradient update step
        rp_policy.update_step(a, b1, b1)

        # prediction step
        rp = rp_policy.predict()
        predicted_rp[i] = rp

    df = pd.DataFrame(bids[:, :2], columns=['b1', 'b2'])
    df['rp'] = predicted_rp
    df['rev_diff'] = np.where(df.b1 < df.rp, -df.b2, np.where(df.b2 > df.rp, 0, df.rp - df.b2))
    print(df.rev_diff.iloc[int(n_rounds*0.1):].sum())

    plt.rc('figure', figsize=(5,2))
    plt.plot(range(n_rounds), bids[:, 0])
    plt.plot(range(n_rounds), bids[:, 1])
    plt.plot(range(n_rounds), predicted_rp)
    plt.show()