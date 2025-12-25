prompt_nolimit_holdem = '''You are now a player in a game of No-limit Texas Hold'em. The game rules are as follows:

1. The deck consists of 52 cards.
2. There are multiple players in the game.
3. Each player is dealt two face-down cards (hole cards).
4. There are five community cards dealt in three stages: the flop (3 cards), the turn (1 card), and the river (1 card).
5. There are four betting rounds: pre-flop, flop, turn, and river.
6. In each round, players can choose to "call", "check", "raise", or "fold".
7. This is a no-limit game, so players can raise any amount from the minimum raise up to their entire stack.
8. The number of raises in each round is unlimited.
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

All possible actions are: "FOLD", "CHECK_CALL", "RAISE_HALF_POT", "RAISE_POT", or "ALL_IN".

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

6. Total chips of the pot:
%s

7. Remaining chips of all players:
%s

8. History actions of all players:
%s

9. Legal actions:
%s

Please tell me your action in JSON format based on the provided information. The JSON should contain an "action" key with a value chose from legal actions. 

Output format examples:
Folding: {"action": "FOLD"}
Checking and calling: {"action": "CHECK_CALL"}
Raising half pot: {"action": "RAISE_HALF_POT"}
Raising pot: {"action": "RAISE_POT"}
Raising all remaining chips: {"action": "ALL_IN"}

Please provide the corresponding JSON action based on the given information.
'''
