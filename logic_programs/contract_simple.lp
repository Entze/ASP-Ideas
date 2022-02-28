
% A contract is a legal binding between two (or more) parties.
party(debtor).
party(creditor).

% The debtor chooses to sign the contract (or they don't).
{ signed_party(debtor);-signed_party(debtor) } = 1.

% The creditor chooses to sign the contract (or they don't).
{ signed_party(creditor);-signed_party(creditor) } = 1.

% Once the creditor and the debtor signed, the creditor is liable to transfer the money, until they do that.
liable_party(creditor) :-
  signed_party(creditor), signed_party(debtor),
  not transfers_party(creditor).

% The creditor may transfer the money (or they don't).
{ transfers_party(creditor);-transfers_party(creditor) } = 1.

% The creditor will not transfer the money if they did not sign the contract.
:- transfers_party(creditor), not signed_party(creditor).
% The creditor will not transfer the money if the debtor did not sign the contract.
:- transfers_party(creditor), not signed_party(debtor).

% If the creditor transferred the money, they are no longer liable.
-liable_party(creditor) :- transfers_party(creditor).

% The debtor becomes liable if both parties signed, the creditor transferred the money and they did not repay the money yet.
liable_party(debtor) :- signed_party(creditor), signed_party(debtor),
  transfers_party(creditor), not transfers_party(debtor).

% The debtor may repay the money (or they don't).
{ transfers_party(debtor);-transfers_party(debtor) } = 1.

% The debtor will not transfer the money if the creditor did not sign the contract.
:- transfers_party(debtor), not signed_party(creditor).
% The debtor will not transfer the money they did not sign the contract.
:- transfers_party(debtor), not signed_party(debtor).
% The debtor will not transfer the money if they didn't receive any money from the creditor.
:- transfers_party(debtor), not transfers_party(creditor).

% If the debtor transferred the money, they are no longer liable.
-liable_party(debtor) :- transfers_party(debtor).

% If the debtor did not sign the contract, the contract may not come into effect
status(disengaged) :- not signed_party(debtor).
% If the creditor did not sign the contract, the contract may not come into effect
status(disengaged) :- not signed_party(creditor).

% If neither party is liable, the contract is fulfilled.
status(fulfilled) :- -liable_party(debtor), -liable_party(creditor).

% If a party is still liable, the contract is violated.
status(violated) :- liable_party(P), party(P).

#show status/1.
#show signed_party/1.
#show liable_party/1.
#show transfers_party/1.
