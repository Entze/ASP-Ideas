
% A13: "d_holdsAt final negation"
-d_holdsAt(X, T + 1) :-
  d_terminates(X, T).

% A14: "-d_holdsAt inertia axiom"
-d_holdsAt(X, T3) :-
  time(T3),
  -d_holdsAt(X, T1), -happens(trigger(X, Delay), T2),
  T1 < T3, T1 <= T2 + Delay, T2 + Delay <= T3.

% "d_clipped definition"
d_clipped(T1, P, T2) :-
  time(T1), time(T2),
  happens(trigger(X, Delay, T)), T1 <= T, T < T2, d_terminates(X, P, T).
