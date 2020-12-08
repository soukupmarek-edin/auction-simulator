import numpy as np


def rp_fee(realtime_kwargs, **params):
    fee = realtime_kwargs['fee']
    winning_bid = realtime_kwargs['winning_bid']
    x0 = realtime_kwargs['x0']
    r = fee * winning_bid + (1 - fee) * x0
    return r


def one_shot(realtime_kwargs, **params):
    r = realtime_kwargs['current_r']
    b1 = realtime_kwargs['winning_bid']
    b2 = realtime_kwargs['second_bid']
    x0 = realtime_kwargs['x0']

    if r == 0:
        r = np.mean([b1, b2])

    if r > b1:
        return np.max([x0, (1-0.3)*r])
    elif r < b2:
        return np.max([x0, (1+0.02)*r])
    else:
        return np.max([x0, (1+0.01)*r])


def gradient_based(realtime_kwargs, **params):
    """
    realtime_kwargs:
    ================
    winning_bid (float)
    second_bid (float)
    current_r (float)

    params:
    =======
    learning_rate (float)
    smoothing_rate (float)
    """
    b1 = realtime_kwargs['winning_bid']
    b2 = realtime_kwargs['second_bid']
    r = realtime_kwargs['current_r']
    x0 = realtime_kwargs['x0']

    eta = params['learning_rate']
    alpha = params['smoothing_rate']

    if b1 == 0 and b2 == 0:
        return 0
    if r == 0:
        return np.mean([b1, b2])

    arr = np.array([b1, b2, r, x0])
    b1, b2, rr, x0 = arr/100

    grad1 = np.exp(-alpha*(rr-b1))/(1+np.exp(-alpha*(rr-b1)))
    grad2 = np.exp(-alpha*(rr-b2))/(1+np.exp(-alpha*(rr-b2)))
    grad3 = -alpha*(b1-x0)*np.exp(-alpha*(rr-b1))/(1+np.exp(-alpha*(rr-b1)))**2
    grad = np.sum([grad1, grad2, grad3])

    r = (rr + eta * grad)*100
    return r
