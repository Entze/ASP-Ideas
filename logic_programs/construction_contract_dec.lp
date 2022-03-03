
party(employer).
party(contractor).

%% TIME

time(1..10).

%% INITIALLY

%!trace_rule {"Employer did not sign the contract intially"}
initiallyFalse(employer_signed_contract).
%!trace_rule {"Contractor did not sign the contract intially"}
initiallyFalse(contractor_signed_contract).

%% Process

initiates(employer_signs_contract, employer_signed_contract, 2).
initiates(contractor_signs_contract, contractor_signed_contract, 2).

initiates(employer_transfers_initial_payment, employer_transferred_initial_payment, 4).
initiates(contractor_receives_initial_payment, contractor_received_initial_payment, 4).

%!trace_rule {"Both parties signed the contract."}
happens(trigger(obligation(achievement, 3, employer_transferred_initial_payment), 0), 3) :-
  holdsAt(employer_signed_contract, 3), holdsAt(contractor_signed_contract, 3).

% Deadlines not working?
%happens(deadline(obligation(achievement, 3, employer_transferred_initial_payment)), 4).
violationTerminable(obligation(achievement, 3, employer_transferred_initial_payment)).
fulfillTerminable(obligation(achievement, 3, employer_transferred_initial_payment)).

happens(trigger(obligation(punctual, 4, contractor_received_initial_payment), 0), 4) :-
  holdsAt(employer_signed_contract, 4), holdsAt(contractor_signed_contract, 4),
  holdsAt(employer_transferred_initial_payment, 4).

%% Story

happens(employer_signs_contract, 2).
happens(contractor_signs_contract, 2).
happens(employer_transfers_initial_payment,4).


%! show_trace holdsAt(employer_transferred_initial_payment, 5).
%!show_trace d_holdsAt(obligation(achievement, 3, employer_transferred_initial_payment), 5).
