% Nurse Scheduling Problem translated from main.py (OR-Tools CP-SAT) to MiniZinc
% -----------------------------------------------------------------------------
% Problem summary:
%  - Assign each of 'num_shifts' shifts per day over 'num_days' to exactly one nurse
%  - Each nurse works at most one shift per day
%  - Total shifts distributed as evenly as possible among nurses:
%       each nurse works between min_per_nurse and max_per_nurse shifts where
%       min_per_nurse = floor(total_shifts / num_nurses)
%       max_per_nurse = min_per_nurse (+1 if division not exact)
%  - Objective: maximize number of requested (preferred) shifts satisfied
%
% Indices are 1-based (MiniZinc convention). Original Python code used 0-based.
%
% Data expected in a separate .dzn file:
%   num_nurses, num_shifts, num_days,
%   array[num_nurses, num_days, num_shifts] of int: shift_requests;
%
% Run example:
%   minizinc nurse_scheduling.mzn nurse_scheduling.dzn
%
% -----------------------------------------------------------------------------

int: num_nurses;
int: num_shifts;        % number of shifts per day
int: num_days;          % scheduling horizon (days)

set of int: NURSES = 1..num_nurses;
set of int: SHIFTS = 1..num_shifts;
set of int: DAYS   = 1..num_days;

% Preference (request) matrix: 1 means nurse requested that shift on that day, 0 otherwise.
array[NURSES, DAYS, SHIFTS] of int: shift_requests;

% Decision variables: x[n,d,s] = true if nurse n works shift s on day d.
array[NURSES, DAYS, SHIFTS] of var bool: x;

% Exactly one nurse per (day, shift).
constraint forall(d in DAYS, s in SHIFTS)(
  sum(n in NURSES)( bool2int(x[n,d,s]) ) = 1
);

% Each nurse works at most one shift per day.
constraint forall(n in NURSES, d in DAYS)(
  sum(s in SHIFTS)( bool2int(x[n,d,s]) ) <= 1
);

% Even distribution of workload across nurses.
int: total_shifts = num_shifts * num_days;
int: min_per_nurse = total_shifts div num_nurses;
int: max_per_nurse = if total_shifts mod num_nurses = 0 then min_per_nurse else min_per_nurse + 1 endif;

constraint forall(n in NURSES)(
  let { int: worked = sum(d in DAYS, s in SHIFTS)( bool2int(x[n,d,s]) ) } in
    worked >= min_per_nurse /\ worked <= max_per_nurse
);

% Objective: maximize fulfilled requests.
var int: objective_value =
  sum(n in NURSES, d in DAYS, s in SHIFTS)( shift_requests[n,d,s] * bool2int(x[n,d,s]) );

solve maximize objective_value;

% Output formatting similar to Python version
output [
  "Objective (requests satisfied) = ", show(objective_value), "\n",
  "Min shifts per nurse = ", show(min_per_nurse), ", Max shifts per nurse = ", show(max_per_nurse), "\n\n",
  concat([
    "Day " ++ show(d) ++ "\n" ++
    concat([
      if fix(x[n,d,s]) then
        "  Nurse " ++ show(n) ++ " works shift " ++ show(s) ++
        (if shift_requests[n,d,s] = 1 then " (requested)\n" else " (not requested)\n" endif)
      else "" endif
      | n in NURSES, s in SHIFTS]) ++ "\n"
    | d in DAYS ])
];
