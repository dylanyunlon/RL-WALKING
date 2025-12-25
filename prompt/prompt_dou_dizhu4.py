prompt_dou_dizhu = '''You are now a player in a game of Dou Dizhu (Fight the Landlord). The game rules are as follows:

1. The game is played by three players with a standard 54-card deck including a red joker and a black joker.
2. There are three roles in the game: landlord, landlord_down (farmer down of landlord), and landlord_up (farmer up of landlord).
3. After bidding, one player becomes the “landlord” who receives an extra three cards. The other two players are the “peasants” who work together to defeat the landlord.
4. In each round, the starting player must play a card or a valid combination of cards.
5. The other two players can choose to either follow with a higher-ranked card or combination, or pass.
6. If two consecutive players pass, the round ends and the player with the highest rank in that round starts the next round.
7. The objective is to be the first player to get rid of all the cards in hand.

The cards and comparison are as follows:
1. Individual cards are ranked. Colored Joker > Black & White Joker > 2 > Ace (A) > King (K) > Queen (Q) > Jack (J) > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3.
2. The Rocket (Red Joker and Black Joker) and the Bomb are groups of cards that work differently in terms of game play.
3. Compare only the same Category. Compare only the Chains with the same length. Compare the rank in the Primal cards only. Jokers and 2 are non-consecutive cards.
4. The type of card combination: Solo, Solo Chain (5), Pair, Pair Chain (3), Trio, Trio Chain (2), Trio with Solo, Trio Chain with Solo, Trio with Pair, Trio Chain with Pair, Bomb, Four with Dual solo, Four with Dual pair.

Your task is to make the best decision in each playing round. I will provide you with the following information:

Turn number:
%s

1. Your role:
%s

2. Your current hand cards:
%s

3. The union of the hand cards of the other two players:
%s

4. The most recent valid move:
%s

5. The played cards so far:
%s

6. The number of cards left for each player:
%s

7. The number of bombs played so far:
%s

8. The historical moves:
%s

9. The legal actions for the current move:
%s

Please tell me what cards you want to play in JSON format based on the provided information. The JSON should contain an "action" key with a value chose from legal actions. 
If you choose to play cards, the value should contain the array of cards you want to play; if you choose to pass, the value should be empty array.

Output format examples: 
Playing cards: {"action": [3, 3, 3]} 
Passing: {"action": []}

Please provide the corresponding JSON action based on the given information.
'''
