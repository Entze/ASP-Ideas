
%% HELPERS

party(creditor).
party(debtor).


%% TIME

time(1..6).


%% FLUENTS

fluent(signed_party(P)) :- party(P).
fluent(liable_party(P)) :- party(P).


%% EVENTS

event(signscontract_party(P)) :- party(P).
event(transfers_party(P)) :- party(P).


%% INITIALLY

initiallyN(liable_party(P)) :- party(P).
initiallyN(signed_party(P)) :- party(P).


%% FLOW

initiates(signscontract_party(P), signed_party(P), T) :- time(T), party(P).

initiates(signscontract_party(debtor), liable_party(creditor), T) :- time(T), holdsAt(signed_party(creditor), T).
initiates(transfers_party(creditor), liable_party(debtor), T) :- time(T), holdsAt(signed_party(debtor), T).

terminates(transfers_party(creditor), liable_party(creditor), T) :- time(T).
terminates(transfers_party(debtor), liable_party(debtor), T) :- time(T).

% This doesn't work, but why?
%trajectory(signed_party(debtor), 4, liable_party(creditor), 5).
%  time(T1), time(T2).
%  T1 < T2.
%  holdsAt(signed_party(creditor), T2).

%% STORY

happens(signscontract_party(creditor), 2).
happens(signscontract_party(debtor), 3).

happens(transfers_party(creditor), 4).
happens(transfers_party(debtor), 5).

%:- holdsAt(liable_party(creditor), 6).
%:- holdsAt(liable_party(debtor), 6).

%% QUERY

% is the creditor liable in the end?
%:- maxtime(T), holdsAt(liable_party(creditor), T).

%% SHOW

#show maxtime/1.
%#show happens/2.
#show holdsAt/2.
%#show -holdsAt/2.
%#show trajectory/4.
