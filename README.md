# auction-simulator

A tool for simulating repeated auctions with multiple objects. The goal is to create an environment that reasonably mimicks real online ad auctions. Such tool would allow testing various policies without having to work with real data, which is often big, difficult to work with or even inaccessible.

In the first stage, the simulator implements tools by which the auction owner may influence the auction. In particular, these are reserve prices and throttling. **Reserve price** is the minimum price the seller is willing to sell their object for. The goal of using reserve prices is mainly to increase the revenue of the auctioneer, but they can also serve other purposes, such as keeping the prices above certain level. **Throttling** means that some bidders are not allowed to participate in certain auctions. Throttling is typically used to solve the problem of early budget depletion - the budgets of some bidders are spent early in the day and thus the impact of the advertising campaign is diminished.

In the second stage, I will develop the behavior of the buyers. The bidder agents may follow more or less sophisticated bidding strategies. The ultimate goal is to implement reinforcement learning techniques which will learn the optimal bidding strategies.

The simulator implements two types of bidders, that correspond to two types of advertising campaigns. **RTB** bidders maximize key performance indicatiors (KPI) subject to budget constraints. **Guaranteed bidders** aim to reach some target KPI at minimum cost.
