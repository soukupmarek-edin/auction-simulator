class ReservePrice:

    @staticmethod
    def rp_fee(house):
        fee = house.auctioned_object.fee
        x0 = house.auctioneer.x0
        r = fee * house.winning_bid + (1 - fee) * x0
        return r
