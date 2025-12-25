prompt_gin_rummy = '''You are now a player in a game of Gin Rummy. The game rules are as follows:

1. The game is played by two players using a standard 52-card deck (ace is low).
2. The dealer deals 11 cards to the opponent and 10 cards to himself.
3. The non-dealer discards first. During each turn, you can pick up the discard or draw from the face-down stock, then discard a card.
4. Players try to form melds of 3 or more cards of the same rank or 3 or more cards of the same suit in sequence.
5. If the deadwood count (the value of non-melded cards) is 10 or less, a player can knock. If all cards can be melded, the player can gin.
6. If a player knocks or gins, the hand ends, and scores are determined. The opponent can lay off deadwood cards to extend melds of the knocker.
7. The score is the difference between the deadwood counts. If the score is positive, the knocker receives it; if zero or negative, the opponent receives the score plus a 25-point undercut bonus.
8. If neither player knocks or gins, they continue drawing and discarding cards. If the stockpile is reduced to two cards, the hand is declared dead.

All possible actions are: "draw_card", "pick_up_discard", "gin", "discard x", "knock x", "declare_dead",  "score N", or "score S".
"draw_card": Draw a card from the stockpile.
"pick_up_discard": Pick up the top card from the discard pile.
"gin": Declare gin.
"discard x": Discard a card from your hand.
"knock x": Knock a card from your hand.
"declare_dead": Declare dead.
"score N": Score player 0.
"score S": Score player 1.

Your task is to make the best decision in each phase of the game. I will provide you with the following information:

Current step:
%s

1. Your id:
%s

2. Your hand cards:
%s
   
3. Top card in the discard pile:
%s

4. Other cards in the discard pile:
%s

5. Opponent known cards:
%s

6. Left card number of stock pile:
%s

7. History actions of all players:
%s

8. Legal actions:
%s

Please tell me your action in JSON format based on the provided information. The JSON should contain an "action" key with a value chose from legal actions.

Output format examples:
Discarding a card: {"action": "discard 3S"}

Please provide the corresponding JSON action based on the given information.
'''
