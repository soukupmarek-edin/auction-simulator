import numpy as np
from AuctionSimulator.AuctionTypes.StandardAuctions import FirstPriceAuction, SecondPriceAuction
from AuctionSimulator.Analysis.Trackers import AuctionTracker, BidderTracker
from AuctionSimulator.AuctionHouse.Throttling import Decision


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

    def __init__(self, n_rounds, auctioneer, bidders, auction_type='second_price',
                 reserve_price_function=None,
                 reserve_price_function_params=None,
                 *,
                 throttling=False, plan=None, probability_function=None, probability_function_params=None,
                 track_auctions=True, track_bidders=True):

        if auction_type == 'second_price':
            self.auction_type = SecondPriceAuction()
        elif auction_type == 'first_price':
            self.auction_type = FirstPriceAuction()
        else:
            raise AttributeError('unknown auction type')

        self.n_rounds = n_rounds
        self.n_bidders = len(bidders)
        self.time = np.linspace(0, 24, n_rounds)
        self.winning_bid = 0
        self.second_bid = 0

        self.reserve_price = 0
        self.reserve_price_function = reserve_price_function
        if reserve_price_function_params is None:
            self.reserve_price_function_params = {}
        else:
            self.reserve_price_function_params = reserve_price_function_params

        self.throttling = throttling
        if throttling:
            self.plan = plan
            self.probability_function = probability_function
            if probability_function_params is None:
                self.probability_function_params = {}
            else:
                self.probability_function_params = probability_function_params

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

        # reserve price
        if self.reserve_price_function:
            realtime_kwargs = {'winning_bid': self.winning_bid,
                               'second_bid': self.second_bid,
                               'fee': auctioned_object.fee,
                               'x0': self.auctioneer.x0[obj_id],
                               'current_r': self.reserve_price}
            self.reserve_price = self.reserve_price_function(realtime_kwargs, **self.reserve_price_function_params)
        else:
            self.reserve_price = self.auctioneer.x0[obj_id]

        # throttling - sets the bids of selected bidders to 0
        if self.throttling:
            realtime_kwargs = {'current_plan': self.plan[self.counter],
                               'current_budgets': np.array([b.budget for b in self.bidders]),
                               'fee': auctioned_object.fee,
                               'bids': bids}
            probabilities = self.probability_function(realtime_kwargs, **self.probability_function_params)
            decisions = Decision.binomial_decision(probabilities)
            bids[decisions] = 0
        else:
            probabilities = np.ones(self.n_bidders)
            decisions = np.ones(self.n_bidders)

        winner, self.winning_bid = self.auction_type.determine_winner(bids, self.reserve_price)
        payment, self.second_bid = self.auction_type.determine_payment(bids, self.reserve_price)

        if winner or winner == 0:
            self.bidders[winner].wins += 1
            self.bidders[winner].budget -= payment
            self.bidders[winner].objects_bought[obj_id] += 1

            self.auctioneer.auctioned_objects[obj_id].quantity -= 1
            self.auctioneer.revenue += payment * (1 - auctioned_object.fee)
            self.auctioneer.fees_paid += payment * auctioned_object.fee
        else:
            self.auctioneer.revenue += self.auctioneer.x0[obj_id]

        if self.auction_tracker:
            self.auction_tracker.data[self.counter, :] = np.array([obj_id, winner, self.winning_bid, self.second_bid,
                                                                   payment, self.reserve_price, auctioned_object.fee])

        if self.bidder_tracker:
            self.bidder_tracker.budgets_data[self.counter, :] = np.array([b.budget for b in self.bidders])
            self.bidder_tracker.bids_data[self.counter, :] = bids
            self.bidder_tracker.probabilities_data[self.counter, :] = probabilities
            self.bidder_tracker.decisions_data[self.counter, :] = decisions

        self.counter += 1
