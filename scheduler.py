import numpy as np
import pandas as pd
from itertools import combinations
from ortools.sat.python import cp_model


def add_shift_assignment_constraints(model, assignments, workers, dates, shifts):
    """ 1. Each shift on each day is assigned to exactly one worker. """
    for date in dates:
        for shift in shifts:
            model.Add(sum(assignments[(worker, date, shift)] for worker in workers) == 1)


def add_worker_daily_shift_constraints(model, assignments, workers, dates, shifts):
    """ 2. Each worker works at most one shift per day. """
    for worker in workers:
        for date in dates:
            model.Add(sum(assignments[(worker, date, shift)] for shift in shifts) <= 1)


def add_no_consecutive_workdays_constraints(model, assignments, workers, dates, shifts):
    """ 3. No consecutive workdays for any worker. """
    for worker in workers:
        for date1, date2 in zip(dates[:-1], dates[1:]):
            model.Add(sum(assignments[(worker, date1, shift)] for shift in shifts) +
                      sum(assignments[(worker, date2, shift)] for shift in shifts) <= 1)


def add_days_off_constraints(model, assignments, dates_off, shifts):
    """ 4. Respect workers' chosen days off according to the dates. """
    for worker, worker_dates in dates_off.items():
        for worker_off_date in worker_dates:
            for shift in shifts:
                model.Add(assignments[(worker, worker_off_date, shift)] == 0)


def add_experts_constraints(model, assignments, experts, dates, shifts, prefer_working):
    for expert in experts:
        for shift in shifts:
            for date in dates:
                if date in prefer_working.get(expert, []):
                    model.Add(assignments[(expert, date, shift)] == int(shift == 1))
                else:
                    model.Add(assignments[(expert, date, shift)] == 0)


def add_balanced_distribution_constraints(model, assignments, workers, dates, shifts, weekday_weights, weighted_freedom,
                                          freedom):
    """ 5. Worker balanced shifts distribution (strict constraint)

    - Ensure workers don't work more than `freedom` shifts than all the others workers.
    - Ensure workers don't work more than `weighted_freedom` weighted_shifts than all the others workers.
    - Ensure workers who work more on weekends work fewer total shifts.
    """

    for worker_1, worker_2 in combinations(workers, 2):
        worker_1_weighted_shifts_count = sum([assignments[(worker_1, date, shift)] * weekday_weights[date.weekday()]
                                              for date in dates for shift in shifts])
        worker_2_weighted_shifts_count = sum([assignments[(worker_2, date, shift)] * weekday_weights[date.weekday()]
                                              for date in dates for shift in shifts])

        worker_1_shifts_count = sum([assignments[(worker_1, date, shift)] for date in dates for shift in shifts])
        worker_2_shifts_count = sum([assignments[(worker_2, date, shift)] for date in dates for shift in shifts])

        model.Add(worker_1_weighted_shifts_count - worker_2_weighted_shifts_count <= weighted_freedom)
        model.Add(worker_1_weighted_shifts_count - worker_2_weighted_shifts_count >= -weighted_freedom)

        model.Add(worker_1_shifts_count - worker_2_shifts_count <= freedom)
        model.Add(worker_1_shifts_count - worker_2_shifts_count >= -freedom)


def add_one_free_weekend_constraints(model, assignments, workers, dates, shifts):
    """ 6. no more than 3 weekends (strict constraint) """
    for worker in workers:
        weekend_dates = [date for date in dates if date.weekday() in [4, 5]]
        worker_shifts_count = sum([assignments[(worker, date, shift)] for date in weekend_dates for shift in shifts])
        model.Add(worker_shifts_count <= 3)


def add_worker_skill_constraints(model, assignments, worker_ratings, non_experts, dates, shifts, max_shift_distance):
    """ 7. Worker skill compatibility (strict constraint)
    - a worker should not do a skill smaller than its skill (4 can do 4-7 etc.)
    - a worker should not do a skill to big than its skill (3 should not do 6 and 7)
    """
    for worker in non_experts:
        worker_rating = worker_ratings[worker]
        for date in dates:
            for shift in shifts:
                if (worker_rating > shift) or (worker_rating + max_shift_distance < shift):
                    model.Add(assignments[(worker, date, shift)] == 0)


def add_shift_rating_penalties(assignments, worker_ratings, non_experts, dates, shifts, penalty_terms, penalty_size):
    """ 8. Prefer workers to be assigned to shifts matching their rating (soft constraint) """
    for worker in non_experts:
        worker_rating = worker_ratings[worker]
        for date in dates:
            for shift in shifts:
                if worker_rating != shift:
                    penalty = abs(worker_rating - shift) * assignments[(worker, date, shift)]
                    penalty_terms.append(penalty * penalty_size)


def add_prefer_not_work_penalties(model, assignments, prefer_not_working, workers, dates, dates_off, shifts,
                                  penalty_terms, penalty_size):
    """ 9. Penalize assigning a worker on a preferred non-working day if other options are available."""
    for worker in workers:
        # Get the days the worker prefers not to work
        worker_prefer_not_working = set(prefer_not_working.get(worker, []))

        # Get the days the worker cannot work
        worker_dates_off = set(dates_off.get(worker, []))

        # Get days the worker has no preference against working
        preferred_dates = list(set(dates) - worker_prefer_not_working - worker_dates_off)

        # Count the days not assigned on preferred days
        preferred_non_assigned_dates_count = sum(
            1 - sum(assignments[(worker, date, shift)] for shift in shifts)
            for date in preferred_dates
        )

        # Count the days assigned on non-preferred days
        assigned_on_non_preferred_count = sum(
            sum(assignments[(worker, date, shift)] for shift in shifts)
            for date in worker_prefer_not_working
        )

        # Define a penalty variable
        penalty_var = model.NewIntVar(0, max(len(dates), len(shifts)), f'{worker}_not_work_penalty')

        # Use `smaller_than` to determine which value to use for the penalty
        smaller_than = model.NewBoolVar(f'{worker}_not_working_smaller_than')
        model.Add(assigned_on_non_preferred_count <= preferred_non_assigned_dates_count).OnlyEnforceIf(smaller_than)
        model.Add(assigned_on_non_preferred_count > preferred_non_assigned_dates_count).OnlyEnforceIf(
            smaller_than.Not())

        # Set penalty_var based on the condition of `smaller_than`
        model.Add(penalty_var == assigned_on_non_preferred_count).OnlyEnforceIf(smaller_than)
        model.Add(penalty_var == preferred_non_assigned_dates_count).OnlyEnforceIf(smaller_than.Not())

        # Add penalty_var to the penalty terms
        penalty_terms.append(penalty_var * penalty_size)


def add_prefer_working_penalties(model, assignments, prefer_working, workers, dates, shifts, penalty_terms,
                                 penalty_size):
    """Penalty for not assigning a worker on a preferred working day while assigning on other non-preferred days."""
    for worker in workers:
        # Get the days the worker prefers to work
        worker_prefer_working = set(prefer_working.get(worker, []))

        # Get days the worker has no preference for working
        non_preferred_dates = list(set(dates) - worker_prefer_working)

        # Count the days not assigned on preferred working days
        not_assigned_on_preferred_count = sum(
            1 - sum(assignments[(worker, date, shift)] for shift in shifts)
            for date in worker_prefer_working
        )

        # Count the days assigned on non-preferred working days
        assigned_on_non_preferred_count = sum(
            sum(assignments[(worker, date, shift)] for shift in shifts)
            for date in non_preferred_dates
        )

        # Define a penalty variable
        penalty_var = model.NewIntVar(0, max(len(dates), len(shifts)), f'{worker}_work_penalty')

        # Use `smaller_than` to determine which value to use for the penalty
        smaller_than = model.NewBoolVar(f'{worker}_working_smaller_than')
        model.Add(assigned_on_non_preferred_count <= not_assigned_on_preferred_count).OnlyEnforceIf(smaller_than)
        model.Add(assigned_on_non_preferred_count > not_assigned_on_preferred_count).OnlyEnforceIf(
            smaller_than.Not())

        # Set penalty_var based on the condition of `smaller_than`
        model.Add(penalty_var == assigned_on_non_preferred_count).OnlyEnforceIf(smaller_than)
        model.Add(penalty_var == not_assigned_on_preferred_count).OnlyEnforceIf(smaller_than.Not())

        # Add penalty_var to the penalty terms
        penalty_terms.append(penalty_var * penalty_size)


def add_shift_distribution_penalties(model, assignments, workers, dates, shifts, penalty_terms, penalty_size):
    """ 11. Fair distribution of total shifts """
    total_shifts_per_worker = len(dates) * len(shifts) // len(workers)
    for worker in workers:
        total_shifts = sum(assignments[(worker, date, shift)] for date in dates for shift in shifts)
        overworked_shifts = model.NewIntVar(0, total_shifts_per_worker, f'overworked_shifts_{worker}')
        model.Add(total_shifts <= total_shifts_per_worker + overworked_shifts)
        penalty_terms.append(penalty_size * overworked_shifts)


def shift_rating_score(df, worker_ratings, workers, dates, shifts, expert_rating, max_shift_distance):
    penalties = 0
    for worker in workers:
        worker_rating = worker_ratings[worker]
        if worker_rating == expert_rating:
            continue
        for date in dates:
            if not np.isnan(df.loc[worker, date]) and worker_rating != df.loc[worker, date]:
                penalties += abs(worker_rating - df.loc[worker, date])
    return 1 - (penalties / (len(shifts) * len(dates) * max_shift_distance))


def prefer_not_work_score(df, prefer_not_working, dates_off, workers, dates):
    all_dates = set(df.columns)
    penalties = 0
    for worker in workers:
        worker_prefer_not_working = prefer_not_working.get(worker, [])
        worker_dates_off = dates_off.get(worker, [])
        preferred_dates = list(all_dates - set(worker_prefer_not_working) - set(worker_dates_off))
        for date in dates:
            if not np.isnan(df.loc[worker, date]) and date in worker_prefer_not_working and len(preferred_dates) != 0:
                penalties += 1
                preferred_dates = preferred_dates[1:]
    return 1 - (penalties / sum([min(len(worker_prefer_not_working), (~df.loc[worker, :].isna()).sum())
                                 for worker, worker_prefer_not_working in prefer_not_working.items()]))


def prefer_work_score(df, prefer_working, dates_off, workers, dates):
    all_dates = set(df.columns)
    penalties = 0
    for worker in workers:
        worker_prefer_working = prefer_working.get(worker, [])
        worker_dates_off = dates_off.get(worker, [])
        other_dates = list(all_dates - set(worker_prefer_working) - set(worker_dates_off))
        for date in dates:
            work_on_date = not np.isnan(df.loc[worker, date])
            date_not_preferable = date in other_dates
            if work_on_date and date_not_preferable and len(worker_prefer_working) != 0:
                penalties += 1
                worker_prefer_working = worker_prefer_working[1:]
    return 1 - (penalties / sum([min(len(worker_prefer_working), (~df.loc[worker, :].isna()).sum())
                                 for worker, worker_prefer_working in prefer_working.items()]))


def shift_distribution_score(df, workers, freedom):
    total_shifts = (~df.isna()).sum(axis=1)
    avg_shifts_per_worker = total_shifts.mean().round().item()
    min_gap = total_shifts.sum() % avg_shifts_per_worker
    penalties = (total_shifts - avg_shifts_per_worker).abs().sum() - min_gap
    return 1 - (penalties / ((freedom / 2) * len(workers)))


def get_weekend_dates(dates):
    """Returns the indices of all Thursdays, Fridays, and Saturdays as weekends."""
    return [date for date in dates if date.weekday() in (3, 4, 5)]


def get_statistics(df_schedule, weekday_weights):
    weekday_weight_series = pd.Series([weekday_weights[date.weekday()] for date in df_schedule.columns],
                                      index=df_schedule.columns)
    statistics_index = ['ראשון עד רביעי', 'חמישי', 'שישי שבת']
    statistics = pd.DataFrame([(~df_schedule.loc[:, weekday_weight_series == value].isna()).sum(axis=1) for value in
                               weekday_weight_series.unique()], index=statistics_index).T

    total_statistics = (~df_schedule.isna()).sum(axis=1)
    total_statistics.name = 'סך הכל'
    statistics = pd.concat([statistics, total_statistics], axis=1)
    return statistics


def get_schedule(source, dates, max_shift_distance, freedom, weighted_freedom, weekday_weights):
    expert_rating = 'בכיר'
    workers = source.name.to_list()
    experts = [row['name'] for index, row in source.iterrows() if row.rating == expert_rating]
    non_experts = list(set(workers) - set(experts))

    # Number of shifts per day
    shifts = range(1, 8)  # Shift numbers will be 1 to 7

    # Worker days off (dates should match actual days off for each worker)
    date_off_string = 'לא יכול/ה'
    dates_off = {source.loc[row, 'name']: source.columns[source.loc[row] == date_off_string].tolist()
                 for row in source.index}

    prefer_not_working_string = 'מעדיפ/ה שלא'
    prefer_not_working = {source.loc[row, 'name']: source.columns[source.loc[row] == prefer_not_working_string].tolist()
                          for row in source.index}
    prefer_not_working = {worker: worker_prefer_not_working
                          for worker, worker_prefer_not_working in prefer_not_working.items() if
                          worker_prefer_not_working}

    prefer_working_string = 'רוצה לעבוד'
    prefer_working = {source.loc[row, 'name']: source.columns[source.loc[row] == prefer_working_string].tolist()
                      for row in source.index}

    # Worker ratings (between 1 and 7)
    worker_ratings = dict(zip(source.name.to_list(), source.rating.to_list()))

    # Create the model
    model = cp_model.CpModel()

    # Variables
    assignments = {}
    for worker in workers:
        for date in dates:
            for shift in shifts:
                assignments[(worker, date, shift)] = model.NewBoolVar(f'shift_{worker}_d{date}_s{shift}')

    # Penalty terms
    penalty_terms = []

    # Add constraints
    add_shift_assignment_constraints(model, assignments, workers, dates, shifts)
    add_worker_daily_shift_constraints(model, assignments, workers, dates, shifts)
    add_no_consecutive_workdays_constraints(model, assignments, non_experts, dates, shifts)

    add_one_free_weekend_constraints(model, assignments, non_experts, dates, shifts)
    add_balanced_distribution_constraints(model, assignments, non_experts, dates, shifts,
                                          weekday_weights, weighted_freedom, freedom)

    add_days_off_constraints(model, assignments, dates_off, shifts)
    add_experts_constraints(model, assignments, experts, dates, shifts, prefer_working)
    add_worker_skill_constraints(model, assignments, worker_ratings, non_experts, dates, shifts, max_shift_distance)

    # Add penalties
    add_shift_rating_penalties(assignments, worker_ratings, non_experts, dates, shifts, penalty_terms, 4)
    add_prefer_not_work_penalties(model, assignments, prefer_not_working, non_experts,
                                  dates, dates_off, shifts, penalty_terms, 1)
    add_prefer_working_penalties(model, assignments, prefer_working, non_experts, dates, shifts, penalty_terms, 1)
    add_shift_distribution_penalties(model, assignments, non_experts, dates, shifts, penalty_terms, 10)

    # Objective: Minimize the sum of all penalty terms
    model.Minimize(sum(penalty_terms))

    # Solving
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Create a DataFrame to hold the results
        schedule_data = {date: [None] * len(workers) for date in dates}

        for date in dates:
            for shift in shifts:
                for worker in workers:
                    if solver.Value(assignments[(worker, date, shift)]):
                        schedule_data[date][workers.index(worker)] = shift

        # Create the DataFrame with the MultiIndex
        df_schedule = pd.DataFrame(schedule_data, index=workers)

        metadata = {'dates_off': dates_off,
                    'prefer_working': prefer_working,
                    'prefer_not_working': prefer_not_working}

        non_experts_prefer_working = {worker: worker_prefer_working for worker, worker_prefer_working
                                      in prefer_working.items() if worker in non_experts}

        scores = {
            'התאמת תורנות לדרגה': shift_rating_score(df_schedule.loc[non_experts], worker_ratings, non_experts,
                                                     dates, shifts, expert_rating, max_shift_distance),
            'העדפה לא לעבוד': prefer_not_work_score(df_schedule.loc[non_experts], prefer_not_working,
                                                    dates_off, non_experts, dates),
            'העדפה לעבוד': prefer_work_score(df_schedule.loc[non_experts], non_experts_prefer_working,
                                             dates_off, non_experts, dates),
            'שוויון בחלוקה': shift_distribution_score(df_schedule.loc[non_experts], non_experts, freedom)
        }

        # Create a list of tuples (worker, rating) for the MultiIndex
        df_schedule.index = pd.MultiIndex.from_tuples([(worker, worker_ratings[worker]) for worker in workers],
                                                      names=['שם', 'דרגה'])

        name_index = df_schedule.index.levels[0].astype(str)
        rating_index = df_schedule.index.levels[1].astype(str)
        df_schedule.index = df_schedule.index.set_levels([name_index, rating_index])

        statistics = get_statistics(df_schedule, weekday_weights)

        return df_schedule, statistics, metadata, scores
    else:
        print("No solution found.")
        return None, None, None, None
