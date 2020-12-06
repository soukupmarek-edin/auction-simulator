class ReservePrice:

    @staticmethod
    def rp_fee(winning_bid, fee, x0):
        r = fee * winning_bid + (1 - fee) * x0
        return r
