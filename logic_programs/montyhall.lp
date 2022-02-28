% HELPERS
door(1..3).
actor(player).
actor(gamemaster).
actor(monty).


% TIME
time(1..6).

% FLUENTS
fluent(prizeisbehind(D)) :- door(D).
fluent(picked(D)) :- door(D).
fluent(opened(D)) :- door(D).
fluent(switched).

% EVENTS
event(choosesdoor_by(D, A)) :- door(D), actor(A).
event(switch).
event(stay).

% STORY

% initially no door is chosen
initiallyN(prizeisbehind(D)) :- door(D).
initiallyN(picked(D)) :- door(D).
initiallyN(openend(D)) :- door(D).

{ happens(choosesdoor_by(D, gamemaster), 2)                                 : door(D) } = 1.
initiates(choosesdoor_by(D, gamemaster), prizeisbehind(D), T) :- time(T), door(D).

{ happens(choosesdoor_by(D, player), 3)                                     : door(D) } = 1.
initiates(choosesdoor_by(D, player), picked(D), T) :- time(T), door(D).

{ happens(choosesdoor_by(D, monty), 4)                                      : door(D) } = 1.
initiates(choosesdoor_by(D, monty), opened(D), T) :- time(T), door(D).

% Monty may only choose doors which are not chosen yet.
:- happens(choosesdoor_by(1, monty), 4), happens(choosesdoor_by(1, gamemaster), 2).
:- happens(choosesdoor_by(2, monty), 4), happens(choosesdoor_by(2, gamemaster), 2).
:- happens(choosesdoor_by(3, monty), 4), happens(choosesdoor_by(3, gamemaster), 2).

:- happens(choosesdoor_by(1, monty), 4), happens(choosesdoor_by(1, player), 3).
:- happens(choosesdoor_by(2, monty), 4), happens(choosesdoor_by(2, player), 3).
:- happens(choosesdoor_by(3, monty), 4), happens(choosesdoor_by(3, player), 3).

{ happens(switch, 5); happens(stay, 5) } = 1.
initiates(switch, switched, 5).
terminates(stay, switched, 5).

% END

won :- maxtime(MT), door(D),
  holdsAt(prizeisbehind(D), MT),
  holdsAt(picked(D),        MT),
  not holdsAt(switched,        MT).
won :- maxtime(MT), door(D1), door(D2),
  D1 != D2,
  holdsAt(prizeisbehind(D1),     MT),
  holdsAt(picked(D2),            MT),
  holdsAt(switched,              MT).
lost :- not won.

% QUERY

% only show models where the player won.
:- lost.

#show happens/2.
#show won/0.
#show lost/0.
