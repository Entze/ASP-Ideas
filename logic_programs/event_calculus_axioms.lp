
% Page 165 (M. Hashmi's Thesis)

starttime(T) :- time(T), not time(T-1).
endtime(T) :- time(T), not time(T+1).

% A1: "holdsAt inertia axiom"
holdsAt(P, T2) :-
  time(T1), time(T2),
  holdsAt(P,T1), T1 < T2, not clipped(T1, P, T2).

% A2:  "holdsAt braking axiom"
-holdsAt(P, T2) :-
  time(T1), time(T2),
  happens(X,T1), terminates(X,P,T1), T1 < T2, not declipped(T1, P, T2).

% A3: "holdsAt momentum axiom"
holdsAt(P, T2) :-
  time(T1), time(T2),
  happens(X, T1), initiates(X,P,T1), T1 < T2, not clipped(T1, P, T2).

% A4: "-holdsAt inertia axiom"
-holdsAt(P, T2) :-
  time(T1), time(T2),
  -holdsAt(P, T1), T1 < T2, not declipped(T1, P, T2).

% A5: "clipped definition"
clipped(T1, P, T2) :-
  time(T1), time(T2),
  happens(X,T), T1 <= T, T < T, terminates(X, P, T).

% A6: "declipped definition"
declipped(T1, P, T2) :-
  time(T1), time(T2),
  happens(X, T), T1 <= T, T < T2, initiates(X, P, T).

% "intializers"

holdsAt(P, ST) :-
  starttime(ST), initiallyTrue(P).

-holdsAt(P, ST) :-
  starttime(ST), initiallyFalse(P).