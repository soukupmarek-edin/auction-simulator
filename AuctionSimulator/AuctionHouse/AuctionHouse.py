import numpy as np
from AuctionSimulator.AuctionTypes.StandardAuctions import FirstPriceAuction, SecondPriceAuction
from AuctionSimulator.Data.Trackers import AuctionTracker, BidderTracker, ObjectTracker
from AuctionSimulator.AuctionHouse.Throttling import Decision
from AuctionSimulator.AuctionHouse.ReservePrice import *
from AuctionSimulator.AuctionHouse.Auctions import Auction


class Controller:
    """
    The main object of the simulator, which assembles all agents and controls the auction. The current auction only
    allows one auctioneer. It also allows multiple bidders.
    In this auction, objects with higher quality are selected for being sold with higher probability.

    Attributes:
    ===========
    N_objects (int): the number of objects in the auction.
    counter (int): the counter of auction rounds.

    Parameters:
    ===========
    auctioneer (object): the auctioneer who wants to sell objects in the auction.
    bidders (array of objects): array of bidder objects.
    auction_type (string): the type of the auction. Must be either 'first_price' or 'second_price'.
    rp_policy (function): a reserve price setting function. The function takes a single argument,
                            the AuctionHouse object.
    throttling (dict): a collection of objects needed to apply a throttling policy: plan, probability function
                    and a decisions function.
    track_auctions (bool): if True, the auction will store auction-specific data with AuctionTracker. Default True.
    track_bidders (bool): if True, the auction will store bidder-specific data with BidderTracker. Default True.

    """

    def __init__(self, n_rounds, auctioneer, bidders,
                 auction_type='second_price',
                 reserve_price_policy=None,
                 track_auctions=False, track_bidders=False, track_objects=False):

        self.auction_type = auction_type
        self.n_rounds = n_rounds

        if not reserve_price_policy:
            self.set_rp = False
        else:
            self.set_rp = True
            self.reserve_price_policy = reserve_price_policy

        self.bidders = bidders
        self.auctioneer = auctioneer
        self.n_objects = self.auctioneer.n_objects
        self.n_bidders = bidders.size

        for bidder in bidders:
            bidder.objects_bought = np.zeros(self.n_objects)

        if track_auctions:
            self.auction_tracker = AuctionTracker(self.n_rounds)
        else:
            self.auction_tracker = None

        if track_bidders:
            self.bidder_tracker = BidderTracker(self.n_rounds, self.n_bidders)
        else:
            self.bidder_tracker = None

        if track_objects:
            self.object_tracker = ObjectTracker(self.n_rounds, self.auctioneer.auctioned_objects[0].n_features)
        else:
            self.object_tracker = None
        self.counter = 0

    def sell_object(self):

        auctioned_object = self.auctioneer.select_object_to_sell()
        assert auctioned_object.quantity > 0, "No more objects to sell"
        bids = np.array([bidder.submit_bid(auctioned_object) for bidder in self.bidders])
        auction = Auction(auctioned_object, bids, self.auction_type)

        # reserve price
        if self.set_rp:
            auction = self.reserve_price_policy.modify_auction(auction)

        # place for throttling

        # tracker update
        if self.auction_tracker:
            self.auction_tracker.data[self.counter, :] = np.array([auctioned_object.id_,
                                                                   auction.sold,
                                                                   auction.winner,
                                                                   auction.winning_bid,
                                                                   auction.second_bid,
                                                                   auction.payment,
                                                                   auction.reserve_price,
                                                                   auctioned_object.fee,
                                                                   auctioned_object.minprice
                                                                   ])

        if self.bidder_tracker:
            self.bidder_tracker.budgets_data[self.counter, :] = np.array([b.budget for b in self.bidders])
            self.bidder_tracker.bids_data[self.counter, :] = bids

        if self.object_tracker:
            self.object_tracker.data[self.counter, :] = np.concatenate([np.array([auctioned_object.id_]), auctioned_object.features])

        self.counter += 1

    def run(self):
        for _ in np.arange(self.n_rounds):
            self.sell_object()
