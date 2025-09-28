"""Monthly nurse scheduling using CP-SAT.

Rules encoded (per user specification):
* 2 shifts: Day (D, 11h) and Night (N, 12h)
* Planning horizon: 4 full weeks (Mon-Sun) + 3 workdays (Mon-Tue-Wed) => 31 days starting on Monday.
  - 23 workdays (Mon-Fri excluding weekends) and 8 weekend days (Saturdays and Sundays within first 4 weeks)
* 19 nurses total
* Workday demand: 7..9 day shifts AND exactly 1 night shift
* Weekend day demand: 5..6 day shifts AND exactly 1 night shift
* Each nurse total hours: 143..146 (Day=11h, Night=12h)
* No 3 consecutive day shifts for any nurse
* Max 3 night shifts per nurse
* Weekend workload balanced: difference in number of weekend shifts between any two nurses <= 1 (implies each has 2 or 3 weekend shifts). Additionally we bound each nurse's weekend shifts in [2,3].
* After a single night shift, the nurse has two free days (within horizon). Pattern D,D,N,off,off allowed.
* Two consecutive night shifts are allowed (N,N) only if there is at least one day off immediately before the pair (if within horizon) and two days off immediately after the pair. Pattern Off,N,N,Off,Off.
* Objective: minimize sum of absolute deviations of nurse hours from 145 plus small penalties for deviation of night distribution and for using extra (above minimum) day shifts.

The model prints a feasible (hopefully optimal) solution and summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import csv
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class ProblemData:
	num_nurses: int = 19
	num_days: int = 31  # 4 weeks (28) + 3 workdays
	day_shift_hours: int = 11
	night_shift_hours: int = 12
	min_hours: int = 143
	max_hours: int = 146
	target_hours: int = 145  # for balancing objective


def build_and_solve(data: ProblemData) -> None:
	model = cp_model.CpModel()

	nurses = range(data.num_nurses)
	days = range(data.num_days)
	# Shift indices
	D, N = 0, 1
	shifts = (D, N)

	# Decision variables x[n][d][s]
	x = {}
	for n in nurses:
		for d in days:
			for s in shifts:
				x[(n, d, s)] = model.new_bool_var(f"x_n{n}_d{d}_{'D' if s==D else 'N'}")

	# Helper lambdas
	def is_weekend(day: int) -> bool:
		# day 0 is Monday; weekend are Saturday(5) & Sunday(6) for only the first 4 weeks (days < 28)
		if day >= 28:  # last 3 workdays (Mon/Tue/Wed)
			return False
		dow = day % 7
		return dow in (5, 6)

	workdays: List[int] = [d for d in days if not is_weekend(d)]
	weekend_days: List[int] = [d for d in days if is_weekend(d)]

	# 1) At most one shift per nurse per day
	for n in nurses:
		for d in days:
			model.add_at_most_one(x[(n, d, s)] for s in shifts)

	# 2) Daily demand constraints
	for d in workdays:
		model.add_linear_constraint(
			sum(x[(n, d, D)] for n in nurses), 7, 9
		)  # day shifts 7..9
		model.add(sum(x[(n, d, N)] for n in nurses) == 1)  # exactly 1 night
	for d in weekend_days:
		model.add_linear_constraint(sum(x[(n, d, D)] for n in nurses), 5, 6)  # 5..6 day shifts
		model.add(sum(x[(n, d, N)] for n in nurses) == 1)

	# 3) Hours per nurse
	hours = []
	deviations = []  # absolute deviations from target
	for n in nurses:
		day_hours = data.day_shift_hours * sum(x[(n, d, D)] for d in days)
		night_hours = data.night_shift_hours * sum(x[(n, d, N)] for d in days)
		h = day_hours + night_hours
		hours.append(h)
		model.add_linear_constraint(h, data.min_hours, data.max_hours)
		# Absolute deviation modeling: h - target = pos - neg, dev = pos + neg
		pos = model.new_int_var(0, data.max_hours - data.target_hours, f"dev_pos_{n}")
		neg = model.new_int_var(0, data.target_hours - data.min_hours, f"dev_neg_{n}")
		model.add(h - data.target_hours == pos - neg)
		dev = model.new_int_var(0, data.max_hours - data.min_hours, f"dev_abs_{n}")
		model.add(dev == pos + neg)
		deviations.append(dev)

	# 4) No 3 consecutive day shifts
	for n in nurses:
		for d in range(data.num_days - 2):
			model.add(sum(x[(n, d + i, D)] for i in range(3)) <= 2)

	# 5) Max 3 night shifts per nurse
	for n in nurses:
		model.add(sum(x[(n, d, N)] for d in days) <= 3)

	# 6) Night shift rest rules with optional consecutive pair:
	# - Single night requires two subsequent off days.
	# - Two consecutive nights (pair) require: previous day off (if exists) and two off days after the pair.
	#   No triple nights allowed.
	for n in nurses:
		# Pair start indicators P_d for nights at d and d+1
		pair_vars = []
		for d in range(data.num_days - 1):
			p = model.new_bool_var(f"pair_n{n}_d{d}")
			pair_vars.append(p)
			# p implies night both days
			model.add(p <= x[(n, d, N)])
			model.add(p <= x[(n, d + 1, N)])
			# If both nights then p =1
			model.add(x[(n, d, N)] + x[(n, d + 1, N)] - 1 <= p)

		# No overlapping pairs (prevents triple nights)
		for d in range(data.num_days - 2):
			model.add(pair_vars[d] + pair_vars[d + 1] <= 1)

		# Singleton night indicators S_d
		singleton = []
		for d in days:
			s = model.new_bool_var(f"single_n{n}_d{d}")
			singleton.append(s)
			# s <= night_d
			model.add(s <= x[(n, d, N)])
			# s + pair covering d must be <=1
			if d - 1 >= 0:
				model.add(s + pair_vars[d - 1] <= 1)
			if d < data.num_days - 1:
				model.add(s + pair_vars[d] <= 1)
			# s >= night_d - pairs around
			left_pair = pair_vars[d - 1] if d - 1 >= 0 else None
			right_pair = pair_vars[d] if d < data.num_days - 1 else None
			parts = [x[(n, d, N)]]
			if left_pair is not None:
				parts.append(-left_pair)
			if right_pair is not None:
				parts.append(-right_pair)
			# Sum(parts) <= s and s <= that expression + (1 - something) etc. Simpler: s >= night - left_pair - right_pair
			# Create linear constraint: night_d - (left_pair if) - (right_pair if) - s <=0
			lin_expr = x[(n, d, N)]
			if left_pair is not None:
				lin_expr = lin_expr - left_pair
			if right_pair is not None:
				lin_expr = lin_expr - right_pair
			model.add(lin_expr - s <= 0)

		# Enforce rest for pairs
		for d in range(data.num_days - 1):
			p = pair_vars[d]
			# Previous day off if exists
			if d - 1 >= 0:
				for sft in shifts:
					model.add(x[(n, d - 1, sft)] + p <= 1)
			# Two days off after pair (d+2, d+3)
			for k in (2, 3):
				if d + k < data.num_days:
					for sft in shifts:
						model.add(x[(n, d + k, sft)] + p <= 1)

		# Enforce rest for singleton nights: two subsequent off days
		for d in days:
			s = singleton[d]
			for k in (1, 2):
				if d + k < data.num_days:
					for sft in shifts:
						model.add(x[(n, d + k, sft)] + s <= 1)

	# 7) Weekend workload balancing (number of weekend shifts per nurse)
	weekend_shifts = []
	for n in nurses:
		w = sum(x[(n, d, s)] for d in weekend_days for s in shifts)
		weekend_shifts.append(w)
		# Each nurse expected 2 or 3 weekend shifts (total feasible range analysis)
		model.add_linear_constraint(w, 2, 3)
	# Pairwise difference <= 1
	for i in nurses:
		for j in nurses:
			if i < j:
				model.add(weekend_shifts[i] - weekend_shifts[j] <= 1)
				model.add(weekend_shifts[j] - weekend_shifts[i] <= 1)

	# 8) (Soft) Minimize number of day shifts above minimums to reduce overstaffing.
	total_day_shifts = sum(x[(n, d, D)] for n in nurses for d in days)
	# Analytical feasible band for day shifts (214..218) but we let constraints decide; optionally clamp if desired.
	# model.add_linear_constraint(total_day_shifts, 214, 218)

	# Objective: minimize (primary) sum of hour deviations + small weight on night imbalance + day shift surplus
	# Night imbalance: variance proxy -> sum of squared difference not linear; approximate by sum |night_count - avg|
	avg_night_times100 = int(100 * (len(days) / data.num_nurses))  # informational only (not used directly)
	night_counts = []
	night_devs = []
	for n in nurses:
		nc = sum(x[(n, d, N)] for d in days)
		night_counts.append(nc)
		# We aim near 1 or 2 nights (since max 3). Target 2 nights for fairness.
		# Deviation from 2 nights.
		pos = model.new_int_var(0, 3, f"night_pos_{n}")
		neg = model.new_int_var(0, 2, f"night_neg_{n}")
		model.add(nc - 2 == pos - neg)
		nd = model.new_int_var(0, 3, f"night_dev_{n}")
		model.add(nd == pos + neg)
		night_devs.append(nd)

	# Surplus day shifts above minimal needed (workday mins + weekend mins)
	min_needed_day_shifts = 7 * len(workdays) + 5 * len(weekend_days)
	surplus = model.new_int_var(0, data.num_days * data.num_nurses, "day_surplus")
	model.add(total_day_shifts - min_needed_day_shifts == surplus)

	# Weights
	W_HOURS = 100  # emphasize meeting target hours tightly
	W_NIGHT_DEV = 10
	W_SURPLUS = 1

	model.minimize(
		W_HOURS * sum(deviations)
		+ W_NIGHT_DEV * sum(night_devs)
		+ W_SURPLUS * surplus
	)

	# Solve
	solver = cp_model.CpSolver()
	solver.parameters.max_time_in_seconds = 60.0
	solver.parameters.num_search_workers = 8
	status = solver.solve(model)

	status_name = solver.status_name(status)
	print(f"Solver status: {status_name}")
	if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		print("No feasible solution found under current constraints.")
		return

	# Report summary
	print("\n=== Summary ===")
	total_hours = 0
	for n in nurses:
		h_val = solver.value(hours[n])
		total_hours += h_val
		day_count = sum(solver.value(x[(n, d, D)]) for d in days)
		night_count = sum(solver.value(x[(n, d, N)]) for d in days)
		wkd = sum(
			solver.value(x[(n, d, s)])
			for d in weekend_days
			for s in shifts
		)
		print(
			f"Nurse {n:2d}: hours={h_val} day_shifts={day_count} night_shifts={night_count} weekend_shifts={wkd}"
		)
	print(f"Total hours (all nurses): {total_hours}")
	print(f"Total day shifts: {solver.value(total_day_shifts)} (min theoretical {min_needed_day_shifts})")
	print(f"Objective value: {solver.objective_value}")

	# Daily roster
	print("\n=== Daily Assignment (D=Day, N=Night) ===")
	for d in days:
		dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d % 7]
		tag = "(WE)" if is_weekend(d) else "   "
		day_workers = [n for n in nurses if solver.boolean_value(x[(n, d, D)])]
		night_worker = [n for n in nurses if solver.boolean_value(x[(n, d, N)])]
		print(
			f"Day {d:02d} {dow} {tag} | D:{len(day_workers)} -> {day_workers} | N:{night_worker}"
		)

	# CSV Export: nurses as rows (1-based), days as columns (1-based)
	csv_filename = "schedule.csv"
	with open(csv_filename, "w", newline="", encoding="utf-8") as f:
		# Use semicolon as requested
		writer = csv.writer(f, delimiter=';')
		header = ["Nurse\\Day"] + [str(d + 1) for d in days]
		writer.writerow(header)
		for n in nurses:
			row = [n + 1]
			for d in days:
				# Per requirement: cell contains 11 (day shift) or 12 (night shift) if nurse works; leave blank if off.
				if solver.boolean_value(x[(n, d, D)]):
					row.append(str(data.day_shift_hours))
				elif solver.boolean_value(x[(n, d, N)]):
					row.append(str(data.night_shift_hours))
				else:
					row.append("")  # Off day
			writer.writerow(row)
	print(f"\nCSV exported to {csv_filename}")

	print("\n=== Solver Stats ===")
	print(f"Conflicts: {solver.num_conflicts}")
	print(f"Branches : {solver.num_branches}")
	print(f"Wall time: {solver.wall_time:.2f}s")


def main() -> None:  # Entry point
	data = ProblemData()
	build_and_solve(data)


if __name__ == "__main__":  # pragma: no cover
	main()
