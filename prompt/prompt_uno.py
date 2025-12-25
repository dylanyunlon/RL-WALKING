prompt_uno = '''You are now a player in a game of UNO. The game rules are as follows:

1. The game is played with a specially designed deck.
2. There are 2 players in the game.
3. Each player starts with seven cards dealt face down.
4. The top card from the Draw Pile is placed in the Discard Pile to start the game.
5. Players take turns matching the card in the Discard Pile by number, color, or symbol/action.
6. If a player has no matching card, they must draw a card from the Draw Pile.
7. If the drawn card can be played, the player must play it; otherwise, they keep the card.
8. The objective is to be the first player to get rid of all the cards in hand.

The deck of UNO includes 108 cards: 
25 in each of four color suits (red, yellow, green, blue), each suit consisting of one zero, two each of 1 through 9, and two each of the action cards "Skip", "Draw Two", and "Reverse". 
The deck also contains four "Wild" cards and four "Wild Draw Four".

Action or Wild cards have the following effects:
- Skip: Next player in the sequence misses a turn.
- Draw Two: Next player in the sequence draws two cards and misses a turn.
- Reverse: Order of play switches directions.
- Wild: Player declares the next color to be matched.
- Wild Draw Four: Player declares the next color to be matched; next player in sequence draws four cards and misses a turn.

Your task is to make the best decision on your turn. I will provide you with the following information:

Current step:
%s

1. Your position:
%s

2. Your hand: 
%s

3. The top card in the Discard Pile: 
%s

4. Played_cards:
%s

5. Number of cards left for each player:
%s

6. History actions of all players:
%s

7. Legal actions:
%s

Please tell me your action in JSON format based on the provided information. The JSON should contain an "action" key with a value chose from legal actions.
The value should be an card or "draw" if you want to draw a card.

Output format examples:
Drawing a card: {"action": "draw"}

Please provide the corresponding JSON action based on the given information.
'''
