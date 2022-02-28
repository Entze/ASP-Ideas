% as described in https://arxiv.org/abs/2106.14566 page 14 (explanaitions) and page 17 (code)

mintime(T) :- time(T), not time(T-1).
maxtime(T) :- time(T), not time(T+1).


%% BEC1
stoppedIn(T1, F, T2) :-
  time(T1), time(T2), time(T),
  fluent(F),
  event(E),
  T1 < T, T < T2,
  terminates(E, F, T),
  happens(E, T).
  
stoppedIn(T1, F, T2) :-
  time(T1), time(T2), time(T),
  fluent(F),
  event(E),
  T1 < T, T < T2,
  releases(E, F, T),
  happens(E, T).


%% BEC2
startedIn(T1, F, T2) :-
  time(T1), time(T2), time(T),
  fluent(F),
  event(E),
  T1 < T, T < T2,
  initiates(E, F, T),
  happens(E, T).

startedIn(T1, F, T2) :-
  time(T1), time(T2), time(T),
  fluent(F),
  event(E),
  T1 < T, T < T2,
  releases(E, F, T),
  happens(E, T).


%% BEC3
holdsAt(F2, T2) :-
  time(T1), time(T2), time(T),
  fluent(F1), fluent(F2),
  trajectory(F1,T1,F2,T2),
  not stoppedIn(T1,F1,T2).


%% BEC4
holdsAt(F, T) :-
  time(T), mintime(MT),
  fluent(F),
  initiallyP(F),
  not stoppedIn(MT, F, T).
  
-holdsAt(F, T) :-
  time(T), mintime(MT),
  fluent(F),
  initiallyN(F),
  not startedIn(MT, F, T).


%% BEC6
holdsAt(F,T2) :-
  time(T1), time(T2),
  fluent(F),
  event(E),
  T1 < T2,
  initiates (E,F,T1),
  happens(E,T1),
  not stoppedIn (T1,F,T2).


%% BEC7
-holdsAt(F,T2) :-
  time(T1), time(T2),
  fluent(F),
  event(E),
  T1 < T2,
  terminates(E,F,T1),
  happens(E,T1),
  not startedIn (T1,F,T2).


%% Consistency
:- -holdsAt(F,T), holdsAt(F,T).  
