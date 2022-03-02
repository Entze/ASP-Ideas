
party(creditor).
party(debtor).

%% TIME

time(1..6).

%% EVENTS

event(creditor_signs_contract).
event(debtor_signs_contract).

%% INITIALLY

initiallyFalse(creditor_signed_contract).
initiallyFalse(debtor_signed_contract).

%% PROCESS

initiates(creditor_signs_contract, creditor_signed_contract, T) :- time(T).
initiates(debtor_signs_contract, debtor_signed_contract, T) :- time(T).

happens(trigger(obligation(punctual, T, creditor_lend), 0), T) :-
  time(T1), time(T2), time(T),
  T1 <= T2, T = T2,
  holdsAt(creditor_signed_contract, T1), holdsAt(debtor_signed_contract, T2),
  not holdsAt(creditor_signed_contract, T1-1), not holdsAt(debtor_signed_contract, T2-1).

happens(trigger(obligation(punctual, T, creditor_lend), 0), T) :-
  time(T1), time(T2), time(T),
  T1 > T2, T = T1,
  holdsAt(creditor_signed_contract, T1), holdsAt(debtor_signed_contract, T2),
  not holdsAt(creditor_signed_contract, T1-1), not holdsAt(debtor_signed_contract, T2-1).

happens(trigger(obligation(punctual, T, debtor_payback), 0), T) :-
  time(TO),
  not happens(violation(obligation(punctual,TO,creditor_lend)),TO),
  d_holdsAt(obligation(punctual, TO, creditor_lend), T-1),
  not d_holdsAt(obligation(punctual, TO, creditor_lend), T).

%% STORY

happens(creditor_signs_contract, 2).
happens(debtor_signs_contract, 2).
{ happens(creditor_lend, 3); -happens(creditor_lend, 3)} = 1 .
{ happens(debtor_payback, 4); -happens(debtor_lend, 4)} = 1 :- happens(creditor_lend, 3).

%#show holdsAt/2.
#show happens/2.