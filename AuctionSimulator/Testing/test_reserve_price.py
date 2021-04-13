import numpy as np
import matplotlib.pyplot as plt
from AuctionSimulator.AuctionHouse.ReservePrice import Appnexus
import pandas as pd

n_rounds = 1000
n_bidders = 50
x0 = 0
bids = np.random.lognormal(mean=5, sigma=1, size=(n_rounds, n_bidders))
bids = np.sort(bids, axis=1)[:, ::-1]

rp_policy = Appnexus(alpha=10, eta=0.05, beta_init=np.array([[1.]]))

lower = 0
upper = 1

if __name__ == '__main__':
    predicted_rp = np.zeros(n_rounds)
    for i, a in enumerate(bids):
        b1 = a[0]
        b2 = a[1]

        lower = lower + 1/(i+1)*(min(lower, b2)-lower)
        upper = upper + 1/(i+1)*(max(upper, b1)-upper)

        # make feature vector
        x = np.array([[1.]])
        b1, b2 = (np.array([b1, b2])-lower)/(upper-lower)

        # gradient update step
        rp_policy.update_step(x, b1, b2, x0)

        # prediction step
        rp = rp_policy.predict(x)
        rp = rp*(upper-lower)+lower
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