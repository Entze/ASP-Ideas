
% A13: "d_holdsAt final negation"
-d_holdsAt(X, TL) :-
  time(TL), time(T),
  TL > T,
  d_terminates(X, T).

% A14: "-d_holdsAt inertia axiom"
-d_holdsAt(X, Tk) :-
  time(Tk),
  -d_holdsAt(X, T), -happens(trigger(X, N), Tj),
  T < Tk, T <= Tj + N, Tj + N <= Tk.

% "d_clipped definition"
d_clipped(T1, P, T2) :-
  time(T1), time(T2),
  happens(trigger(X, N), T), T1 <= T, T < T2, d_terminates(X, P, T).
