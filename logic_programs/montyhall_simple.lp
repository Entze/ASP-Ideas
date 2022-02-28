
door(1..3).

actor(gamemaster).
actor(player).
actor(monty).

{ doorchosen_by(D, A) : door(D) } = 1 :- actor(A).

:- doorchosen_by(1, monty), doorchosen_by(1, gamemaster).
:- doorchosen_by(2, monty), doorchosen_by(2, gamemaster).
:- doorchosen_by(3, monty), doorchosen_by(3, gamemaster).

:- doorchosen_by(1, monty), doorchosen_by(1, player).
:- doorchosen_by(2, monty), doorchosen_by(2, player).
:- doorchosen_by(3, monty), doorchosen_by(3, player).

{ switch; stay } = 1.

won :- door(D),                      
  doorchosen_by(D, gamemaster), doorchosen_by(D, player),
  stay.
won :- door(D1), door(D2), D1 != D2,
  doorchosen_by(D1, gamemaster), doorchosen_by(D2, player),
  switch.

lost :- not won.
