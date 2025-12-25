prompt_limit_holdem = '''You are now a player in a game of Limit Texas Hold'em. The game rules are as follows:

1. The deck consists of 52 cards.
2. There are multiple players in the game.
3. Each player is dealt two face-down cards (hole cards).
4. There are five community cards dealt in three stages: the flop (3 cards), the turn (1 card), and the river (1 card).
5. There are four betting rounds: pre-flop, flop, turn, and river.
6. In each round, players can choose to "call", "check", "raise", or "fold".
7. This is a fixed limit game, so raises are of a fixed amount.
8. The number of raises in each round is limited to 4.
9. The winner is determined by the best five-card hand using any combination of hole cards and community cards.

Texas Hold'em hands are ranked from highest to lowest as follows: 
Royal Flush: A, K, Q, J, 10 all of the same suit.
Straight Flush: Five consecutive cards of the same suit. Higher top card wins.
Four of a Kind: Four cards of the same rank. Higher rank wins; if same, compare fifth card.
Full House: Three cards of one rank and two cards of another rank. Higher three-card rank wins; if same, compare the two-card rank.
Flush: Five non-consecutive cards of the same suit. Compare the highest card, then the second-highest, and so on.
Straight: Five consecutive cards of different suits. Higher top card wins.
Three of a Kind: Three cards of the same rank. Higher rank wins.
Two Pair: Two cards of one rank and two cards of another rank. Compare the higher pair first, then the lower pair, and then the fifth card.
One Pair: Two cards of the same rank. Compare the pair first, then the highest non-paired card, then the second highest, and so on.
High Card: If no hand can be formed, the highest card wins. If the highest cards are the same, compare the second highest, and so on.
If the hands are of equal rank, the pot is split.

All possible actions are: "fold", "call", "raise", or "check".

Your task is to make the best decision in each betting round. I will provide you with the following information:

Current betting round:
%s

1. Your position:
%s

2. Your hole cards:
%s

3. Community cards:
%s

4. Your chips in the pot:
%s

5. All chips in the pot:
%s

6. Number of raises so far in four rounds: 
%s

7. History actions of all players:
%s

8. Legal actions:
%s

Please tell me your action in JSON format based on the provided information. The JSON should contain an "action" key with a value chose from legal actions. 

Output format examples: 
Folding: {"action": "fold"}
Calling: {"action": "call"}
Raising: {"action": "raise"}
Checking: {"action": "check"}

Please provide the corresponding JSON action based on the given information.
'''
