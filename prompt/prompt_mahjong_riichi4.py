prompt_riichi = '''You are now a player in a game of Riichi Mahjong. The game rules are as follows:

1. The game uses 136 tiles divided into three suits (Pin, Sou, Wan) and honor tiles, which include wind and dragon tiles.
2. The tiles are mixed and arranged into four walls, two tiles high and 17 tiles wide.
3. Players draw and discard tiles to form valid groups (mentsu) of triplets (Pon), sequences (Chii), or quads (Kan).
4. A hand can be completed to declare a win by forming four groups and a pair.
5. Special rules include Riichi (declaring ready with a closed hand) and Dora indicators (bonus tiles).
6. Players can call tiles discarded by others to make open sets, making their hands open or closed.

All possible actions are: 'dahai: x', 'reach', 'chi_low', 'chi_mid', 'chi_high', 'pon', 'kan', 'hora', 'ryukyoku', 'pass'.
'dahai: x': discard tile x.
'reach': declare a ready hand (riichi).
'chi_low', 'chi_mid', 'chi_high': Create a meld by completing a sequence, using the discarded tile.
'pon': create a three-of-a-kind meld using the discarded tile.
'kan': create a four-of-a-kind meld. This can be done in several ways: by adding a tile to an existing three-of-a-kind meld, using a discarded tile to complete a four-of-a-kind, or declaring a concealed four-of-a-kind by having four identical tiles in hand.
'hora': declare a win.
'ryukyoku': declare an aborted game or a draw. 
'pass': Opt not to take any action or declaration. This can mean passing on a chance to chi, pon, kan, or win (hora).

Your task is to make the best decision in each playing round. I will provide you with the following information:

1. Your identifier:
%s

2. bakaze:
%s

3. jikaze:
%s

4. kyoku:
%s

5. honba:
%s

6. kyotaku:
%s

7. oya:
%s

8. Scores:
%s

9. Your rank:
%s

10. at turn:
%s

11. title left:
%s

12. shanten:
%s

13. my hands:
%s

14. wait tiles:
%s

15. dora indicators:
%s

16. dora owned:
%s

17. akas in your hand:
%s

18. doras seen:
%s

19. akas seen:
%s

20. tiles seen:
%s

21. ankan candidates:
%s

22. kakan candidates:
%s

23. kawa overview:
%s

24. fuuro overview:
%s

25. ankan overview:
%s

26. last tedashis:
%s

27. riichi sutehais:
%s

28. last self tsumo:
%s

29. last kawa tile:
%s

30. riichi declared:
%s

31. riichi accepted:
%s

32. can riichi:
%s

33. is riichi:
%s

34. at furiten:
%s

35. is menzen:
%s

36. Legal actions:
%s

Please tell me your action in JSON format based on the provided information. The JSON should contain an "action" key with a value chose from legal actions.

Output format examples:
Playing a card: {"action": "dahai: x"}

Please provide the corresponding JSON action based on the given information.
'''
