prompt_leduc_holdem = '''You are now a player in a game of Leduc Hold'em. The game rules are as follows:

1. The deck consists of only two pairs of King, Queen and Jack (6 cards in total).
2. There are two players in the game.
3. The game has two rounds with a two-bet maximum.
4. Raise amounts are 2 in the first round and 4 in the second round.
5. In the first round, each player puts 1 unit in the pot and is dealt one card.
6. In the second round, one public card is revealed.
7. The winner is determined by matching the player's card with the public card or having the highest rank.

All possible actions are: "fold", "call", "raise", or "check".

Your task is to make the best decision in each betting round. I will provide you with the following information:

Round number:
%s

1. Your position:
%s

2. Your hand:
%s

3. Public card  (if in round 2):
%s

4. Your chips in the pot:
%s

5. All chips in the pot:
%s

6. Number of raises so far in two rounds: 
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
