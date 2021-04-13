import numpy as np
from AuctionSimulator.AuctionTypes.StandardAuctions import FirstPriceAuction, SecondPriceAuction
from AuctionSimulator.Analysis.Trackers import AuctionTracker, BidderTracker
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
                 throttling=False, plan=None, probability_function=None,
                 track_auctions=True, track_bidders=True):

        self.auction_type = auction_type
        self.n_rounds = n_rounds
        self.n_bidders = len(bidders)
        self.time = np.linspace(0, 24, n_rounds)
        self.winning_bid = 0
        self.second_bid = 0

        self.rp = 0
        if reserve_price_policy is None:
            self.reserve_price_policy = Basic()
        else:
            self.reserve_price_policy = reserve_price_policy

        self.throttling = throttling
        if throttling:
            self.plan = plan
            self.probability_function = probability_function

        self.bidders = bidders
        self.auctioneer = auctioneer
        self.n_objects = self.auctioneer.n_objects

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
        self.counter = 0

    def run(self):
        for _ in np.arange(self.n_rounds):
            self.sell_object()

    def sell_object(self):
        """
        The process of selling the chosen object. The process goes as follows:
        1) all bidders submit their bids.
        2) reserve price is set.
        3) winner and winning bid are determined according to the chosen rules.
        4) payment is determined according to the chosen rules.
        5) if there's a winner in the auction:
            5.1) the winner is added a win to the counter of wins, deducted the payment from the budget and added the
                object to the counter of bought objects.
            5.2) the quantity of the sold object is decreased by one.
            5.3) the auctioneer's revenue is increased by the payment*(1-fee) and their fees total is increased
                by payment*fee.
        6) if the either of the tracker objects are assigned to the auction, the requested values are recorded.
        7) the round counter attribute is increased by one.

        """
        auctioned_object = self.auctioneer.select_object_to_sell()
        obj_id = auctioned_object.id_
        assert auctioned_object.quantity > 0, "No more objects to sell"
        bids = np.array([bidder.submit_bid(auctioned_object) for bidder in self.bidders])
        auction = Auction(auctioned_object, bids, self.auction_type)

        # reserve price
        auction = self.reserve_price_policy.modify_auction(auction)
        rp = self.reserve_price_policy.rp

        # throttling - sets the bids of throttled bidders to 0
        if self.throttling:
            realtime_kwargs = {'current_plan': self.plan[self.counter],
                               'current_budgets': np.array([b.budget for b in self.bidders]),
                               'fee': auctioned_object.fee,
                               'bids': bids}
            probabilities = self.probability_function(realtime_kwargs)
            decisions = Decision.binomial_decision(probabilities)
            bids[decisions] = 0
        else:
            probabilities = np.ones(self.n_bidders)
            decisions = np.ones(self.n_bidders)

        winner, winning_bid = auction.winner, auction.winning_bid
        second_bid = auction.second_bid
        payment = auction.payment
        revenue = auction.revenue

        # data update
        self.bidders[winner].wins += auction.sold
        self.bidders[winner].budget -= payment
        self.bidders[winner].objects_bought[obj_id] += auction.sold

        self.auctioneer.auctioned_objects[obj_id].quantity -= auction.sold
        self.auctioneer.revenue += revenue
        self.auctioneer.fees_paid += payment * auctioned_object.fee * auction.sold

        # trackers
        if self.auction_tracker:
            self.auction_tracker.data[self.counter, :] = np.array([obj_id, winner, winning_bid, second_bid,
                                                                   payment, rp, auctioned_object.fee, auctioned_object.x0])

        if self.bidder_tracker:
            self.bidder_tracker.budgets_data[self.counter, :] = np.array([b.budget for b in self.bidders])
            self.bidder_tracker.bids_data[self.counter, :] = bids
            self.bidder_tracker.probabilities_data[self.counter, :] = probabilities
            self.bidder_tracker.decisions_data[self.counter, :] = decisions

        self.counter += 1
