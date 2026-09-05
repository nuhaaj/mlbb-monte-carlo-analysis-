import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEED = 42

BASE_WIN_PROBABILITY = 0.55
MATCHES_PER_SEASON = 200
NUMBER_OF_SIMULATIONS = 10000

TARGET_STARS = 50

WIN_PROBABILITIES = [
    0.45,
    0.50,
    0.525,
    0.55,
    0.575,
    0.60,
    0.65
]

HORIZONS = [
    100,
    200,
    300,
    500
]

OUTPUT_FOLDER = "MLBB_Monte_Carlo_Output"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


rng = np.random.default_rng(SEED)


def longest_streak(results, target):

    longest = 0
    current = 0

    for result in results:

        if result == target:

            current += 1

            longest = max(
                longest,
                current
            )

        else:

            current = 0

    return longest



def wilson_interval(
    successes,
    total,
    z=1.96
):

    if total == 0:

        return np.nan, np.nan

    p = successes / total

    denominator = (
        1
        + z**2 / total
    )

    centre = (
        p
        + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        / denominator
        * np.sqrt(
            (
                p * (1 - p)
                / total
            )
            +
            (
                z**2
                / (4 * total**2)
            )
        )
    )

    lower = centre - margin
    upper = centre + margin

    return lower, upper


def simulate_match_seasons(
    probability,
    number_of_matches,
    number_of_simulations,
    rng
):


    outcomes = (
        rng.random(
            (
                number_of_simulations,
                number_of_matches
            )
        )
        < probability
    )

    wins = outcomes.sum(
        axis=1
    )

    observed_win_rates = (
        wins
        / number_of_matches
    )

    longest_wins = []
    longest_losses = []

    for row in outcomes:

        longest_wins.append(
            longest_streak(
                row,
                True
            )
        )

        longest_losses.append(
            longest_streak(
                row,
                False
            )
        )

    simulations = pd.DataFrame({

        "Simulation":
            np.arange(
                1,
                number_of_simulations + 1
            ),

        "Wins":
            wins,

        "Losses":
            number_of_matches - wins,

        "Observed_Win_Rate":
            observed_win_rates,

        "Longest_Win_Streak":
            longest_wins,

        "Longest_Loss_Streak":
            longest_losses
    })

    return simulations


def simulate_rank_journeys(
    probability,
    number_of_matches,
    number_of_simulations,
    target_stars,
    rng
):

    outcomes = (
        rng.random(
            (
                number_of_simulations,
                number_of_matches
            )
        )
        < probability
    )


    star_changes = np.where(
        outcomes,
        1,
        -1
    )

    star_paths = np.cumsum(
        star_changes,
        axis=1
    )

    simulation_rows = []

    for simulation in range(
        number_of_simulations
    ):

        path = star_paths[
            simulation
        ]

        match_results = outcomes[
            simulation
        ]

        target_locations = np.flatnonzero(
            path >= target_stars
        )

        if len(target_locations) > 0:

            target_reached = True

            matches_to_target = (
                target_locations[0]
                + 1
            )

            stop_match = (
                matches_to_target
            )

        else:

            target_reached = False

            matches_to_target = np.nan

            stop_match = (
                number_of_matches
            )

        journey_path = path[
            :stop_match
        ]

        journey_results = match_results[
            :stop_match
        ]

        journey_with_start = np.concatenate(
            (
                [0],
                journey_path
            )
        )

        running_peak = np.maximum.accumulate(
            journey_with_start
        )

        drawdowns = (
            running_peak
            - journey_with_start
        )

        maximum_drawdown = (
            drawdowns.max()
        )

        longest_loss_to_stop = longest_streak(
            journey_results,
            False
        )

        simulation_rows.append({

            "Simulation":
                simulation + 1,

            "Target_Reached":
                target_reached,

            "Matches_to_Target":
                matches_to_target,

            "Terminal_Stars_After_Horizon":
                path[-1],

            "Highest_Star_Position":
                path.max(),

            "Lowest_Star_Position":
                path.min(),

            "Maximum_Drawdown_To_Stop":
                maximum_drawdown,

            "Longest_Loss_Streak_To_Stop":
                longest_loss_to_stop
        })

    return pd.DataFrame(
        simulation_rows
    )


single_results = (
    rng.random(
        MATCHES_PER_SEASON
    )
    < BASE_WIN_PROBABILITY
)

single_wins = (
    single_results.sum()
)

single_losses = (
    MATCHES_PER_SEASON
    - single_wins
)

single_wr = (
    single_wins
    / MATCHES_PER_SEASON
)

single_longest_win = longest_streak(
    single_results,
    True
)

single_longest_loss = longest_streak(
    single_results,
    False
)


print("\n========================================")
print("ONE SIMULATED 200-MATCH SEASON")
print("========================================")

print(
    "True win probability:",
    round(
        BASE_WIN_PROBABILITY * 100,
        2
    ),
    "%"
)

print(
    "Wins:",
    single_wins
)

print(
    "Losses:",
    single_losses
)

print(
    "Observed win rate:",
    round(
        single_wr * 100,
        2
    ),
    "%"
)

print(
    "Longest winning streak:",
    single_longest_win
)

print(
    "Longest losing streak:",
    single_longest_loss
)


baseline_match_simulations = (
    simulate_match_seasons(
        BASE_WIN_PROBABILITY,
        MATCHES_PER_SEASON,
        NUMBER_OF_SIMULATIONS,
        rng
    )
)


print("\n========================================")
print("BASELINE MONTE CARLO")
print("========================================")

print(
    baseline_match_simulations.head()
)

print(
    "\nShape:",
    baseline_match_simulations.shape
)


average_observed_wr = (
    baseline_match_simulations[
        "Observed_Win_Rate"
    ].mean()
)

observed_wr_std = (
    baseline_match_simulations[
        "Observed_Win_Rate"
    ].std()
)

average_longest_loss = (
    baseline_match_simulations[
        "Longest_Loss_Streak"
    ].mean()
)

median_longest_loss = (
    baseline_match_simulations[
        "Longest_Loss_Streak"
    ].median()
)

loss_streak_90th = np.percentile(
    baseline_match_simulations[
        "Longest_Loss_Streak"
    ],
    90
)

loss_streak_95th = np.percentile(
    baseline_match_simulations[
        "Longest_Loss_Streak"
    ],
    95
)

prob_5_loss = (
    baseline_match_simulations[
        "Longest_Loss_Streak"
    ] >= 5
).mean()

prob_7_loss = (
    baseline_match_simulations[
        "Longest_Loss_Streak"
    ] >= 7
).mean()

prob_10_loss = (
    baseline_match_simulations[
        "Longest_Loss_Streak"
    ] >= 10
).mean()


print("\n========================================")
print("BASELINE STREAK RISK")
print("========================================")

print(
    "Average observed WR:",
    round(
        average_observed_wr * 100,
        2
    ),
    "%"
)

print(
    "Average longest losing streak:",
    round(
        average_longest_loss,
        2
    )
)

print(
    "Median longest losing streak:",
    median_longest_loss
)

print(
    "90th percentile losing streak:",
    loss_streak_90th
)

print(
    "95th percentile losing streak:",
    loss_streak_95th
)

print(
    "P(5+ loss streak):",
    round(
        prob_5_loss * 100,
        2
    ),
    "%"
)

print(
    "P(7+ loss streak):",
    round(
        prob_7_loss * 100,
        2
    ),
    "%"
)

print(
    "P(10+ loss streak):",
    round(
        prob_10_loss * 100,
        2
    ),
    "%"
)

theoretical_wr_mean = (
    BASE_WIN_PROBABILITY
)

theoretical_wr_standard_error = np.sqrt(
    (
        BASE_WIN_PROBABILITY
        *
        (
            1
            - BASE_WIN_PROBABILITY
        )
    )
    /
    MATCHES_PER_SEASON
)


print("\n========================================")
print("MODEL VALIDATION: WIN RATE")
print("========================================")

print(
    "Theoretical mean WR:",
    round(
        theoretical_wr_mean * 100,
        2
    ),
    "%"
)

print(
    "Simulated mean WR:",
    round(
        average_observed_wr * 100,
        2
    ),
    "%"
)

print(
    "Theoretical WR standard error:",
    round(
        theoretical_wr_standard_error
        * 100,
        2
    ),
    "percentage points"
)

print(
    "Simulated WR standard deviation:",
    round(
        observed_wr_std * 100,
        2
    ),
    "percentage points"
)


observed_wr_2_5 = np.percentile(
    baseline_match_simulations[
        "Observed_Win_Rate"
    ],
    2.5
)

observed_wr_97_5 = np.percentile(
    baseline_match_simulations[
        "Observed_Win_Rate"
    ],
    97.5
)

prob_wr_below_50 = (
    baseline_match_simulations[
        "Observed_Win_Rate"
    ] < 0.50
).mean()

prob_wr_below_52_5 = (
    baseline_match_simulations[
        "Observed_Win_Rate"
    ] < 0.525
).mean()

prob_wr_at_least_60 = (
    baseline_match_simulations[
        "Observed_Win_Rate"
    ] >= 0.60
).mean()


print("\n========================================")
print("OBSERVED PERFORMANCE UNCERTAINTY")
print("========================================")

print(
    "2.5th percentile observed WR:",
    round(
        observed_wr_2_5 * 100,
        2
    ),
    "%"
)

print(
    "97.5th percentile observed WR:",
    round(
        observed_wr_97_5 * 100,
        2
    ),
    "%"
)

print(
    "P(observed WR < 50%):",
    round(
        prob_wr_below_50 * 100,
        2
    ),
    "%"
)

print(
    "P(observed WR < 52.5%):",
    round(
        prob_wr_below_52_5 * 100,
        2
    ),
    "%"
)

print(
    "P(observed WR >= 60%):",
    round(
        prob_wr_at_least_60 * 100,
        2
    ),
    "%"
)


match_scenario_rows = []

for probability in WIN_PROBABILITIES:

    scenario = simulate_match_seasons(
        probability,
        MATCHES_PER_SEASON,
        NUMBER_OF_SIMULATIONS,
        rng
    )

    loss_streaks = scenario[
        "Longest_Loss_Streak"
    ]

    win_rates = scenario[
        "Observed_Win_Rate"
    ]

    match_scenario_rows.append({

        "True_Win_Rate":
            probability * 100,

        "Average_Observed_WR":
            win_rates.mean() * 100,

        "Observed_WR_2_5th_Pct":
            np.percentile(
                win_rates,
                2.5
            ) * 100,

        "Observed_WR_97_5th_Pct":
            np.percentile(
                win_rates,
                97.5
            ) * 100,

        "P_Observed_WR_Below_50":
            (
                win_rates < 0.50
            ).mean() * 100,

        "Average_Longest_Loss_Streak":
            loss_streaks.mean(),

        "P_5_Plus_Loss_Streak":
            (
                loss_streaks >= 5
            ).mean() * 100,

        "P_7_Plus_Loss_Streak":
            (
                loss_streaks >= 7
            ).mean() * 100,

        "P_10_Plus_Loss_Streak":
            (
                loss_streaks >= 10
            ).mean() * 100,

        "Loss_Streak_95th_Percentile":
            np.percentile(
                loss_streaks,
                95
            )
    })


match_scenario_summary = pd.DataFrame(
    match_scenario_rows
)


print("\n========================================")
print("MATCH / STREAK SCENARIO ANALYSIS")
print("========================================")

print(
    match_scenario_summary
    .round(2)
    .to_string(index=False)
)


single_rank = simulate_rank_journeys(
    BASE_WIN_PROBABILITY,
    MATCHES_PER_SEASON,
    1,
    TARGET_STARS,
    rng
)


print("\n========================================")
print("ONE EXAMPLE RANK JOURNEY")
print("========================================")

print(
    single_rank
    .round(2)
    .to_string(index=False)
)


baseline_rank_simulations = (
    simulate_rank_journeys(
        BASE_WIN_PROBABILITY,
        MATCHES_PER_SEASON,
        NUMBER_OF_SIMULATIONS,
        TARGET_STARS,
        rng
    )
)


successful_rank_journeys = (
    baseline_rank_simulations[
        baseline_rank_simulations[
            "Target_Reached"
        ]
    ]
)

successful_count = len(
    successful_rank_journeys
)

target_probability = (
    successful_count
    / NUMBER_OF_SIMULATIONS
)

target_ci_lower, target_ci_upper = (
    wilson_interval(
        successful_count,
        NUMBER_OF_SIMULATIONS
    )
)

average_terminal_stars = (
    baseline_rank_simulations[
        "Terminal_Stars_After_Horizon"
    ].mean()
)

terminal_star_std = (
    baseline_rank_simulations[
        "Terminal_Stars_After_Horizon"
    ].std()
)

theoretical_terminal_mean = (
    MATCHES_PER_SEASON
    *
    (
        2 * BASE_WIN_PROBABILITY
        - 1
    )
)

theoretical_terminal_sd = np.sqrt(
    4
    *
    MATCHES_PER_SEASON
    *
    BASE_WIN_PROBABILITY
    *
    (
        1
        - BASE_WIN_PROBABILITY
    )
)

average_drawdown = (
    baseline_rank_simulations[
        "Maximum_Drawdown_To_Stop"
    ].mean()
)

drawdown_95th = np.percentile(
    baseline_rank_simulations[
        "Maximum_Drawdown_To_Stop"
    ],
    95
)

prob_10_drawdown = (
    baseline_rank_simulations[
        "Maximum_Drawdown_To_Stop"
    ] >= 10
).mean()

prob_15_drawdown = (
    baseline_rank_simulations[
        "Maximum_Drawdown_To_Stop"
    ] >= 15
).mean()

prob_20_drawdown = (
    baseline_rank_simulations[
        "Maximum_Drawdown_To_Stop"
    ] >= 20
).mean()


if successful_count > 0:

    average_matches_to_target = (
        successful_rank_journeys[
            "Matches_to_Target"
        ].mean()
    )

    median_matches_to_target = (
        successful_rank_journeys[
            "Matches_to_Target"
        ].median()
    )

    p10_matches_to_target = np.percentile(
        successful_rank_journeys[
            "Matches_to_Target"
        ],
        10
    )

    p90_matches_to_target = np.percentile(
        successful_rank_journeys[
            "Matches_to_Target"
        ],
        90
    )

else:

    average_matches_to_target = np.nan

    median_matches_to_target = np.nan

    p10_matches_to_target = np.nan

    p90_matches_to_target = np.nan


print("\n========================================")
print("BASELINE RANK RESULTS")
print("========================================")

print(
    "Successful journeys:",
    successful_count,
    "out of",
    NUMBER_OF_SIMULATIONS
)

print(
    "P(reach +50 stars within 200 matches):",
    round(
        target_probability * 100,
        2
    ),
    "%"
)

print(
    "95% Monte Carlo CI:",
    round(
        target_ci_lower * 100,
        2
    ),
    "% to",
    round(
        target_ci_upper * 100,
        2
    ),
    "%"
)

print(
    "Average terminal stars:",
    round(
        average_terminal_stars,
        2
    )
)

print(
    "Theoretical expected terminal stars:",
    round(
        theoretical_terminal_mean,
        2
    )
)

print(
    "Simulated terminal-star SD:",
    round(
        terminal_star_std,
        2
    )
)

print(
    "Theoretical terminal-star SD:",
    round(
        theoretical_terminal_sd,
        2
    )
)

print(
    "Average maximum drawdown:",
    round(
        average_drawdown,
        2
    )
)

print(
    "95th percentile maximum drawdown:",
    drawdown_95th
)

print(
    "P(10+ star drawdown):",
    round(
        prob_10_drawdown * 100,
        2
    ),
    "%"
)

print(
    "P(15+ star drawdown):",
    round(
        prob_15_drawdown * 100,
        2
    ),
    "%"
)

print(
    "P(20+ star drawdown):",
    round(
        prob_20_drawdown * 100,
        2
    ),
    "%"
)

print(
    "Average matches to target:",
    round(
        average_matches_to_target,
        2
    )
    if successful_count > 0
    else "N/A"
)

print(
    "Median matches to target:",
    median_matches_to_target
    if successful_count > 0
    else "N/A"
)


seven_plus_streak = (
    baseline_rank_simulations[
        "Longest_Loss_Streak_To_Stop"
    ] >= 7
)

no_seven_plus_streak = (
    ~seven_plus_streak
)


if seven_plus_streak.sum() > 0:

    target_given_7_plus = (
        baseline_rank_simulations.loc[
            seven_plus_streak,
            "Target_Reached"
        ].mean()
    )

else:

    target_given_7_plus = np.nan


if no_seven_plus_streak.sum() > 0:

    target_without_7_plus = (
        baseline_rank_simulations.loc[
            no_seven_plus_streak,
            "Target_Reached"
        ].mean()
    )

else:

    target_without_7_plus = np.nan


print("\n========================================")
print("STREAK RISK VS TARGET ATTAINMENT")
print("========================================")

print(
    "P(reach target | 7+ loss streak):",
    round(
        target_given_7_plus * 100,
        2
    ),
    "%"
)

print(
    "P(reach target | no 7+ loss streak):",
    round(
        target_without_7_plus * 100,
        2
    ),
    "%"
)


rank_scenario_rows = []

for probability in WIN_PROBABILITIES:

    rank_scenario = (
        simulate_rank_journeys(
            probability,
            MATCHES_PER_SEASON,
            NUMBER_OF_SIMULATIONS,
            TARGET_STARS,
            rng
        )
    )

    successful = (
        rank_scenario[
            rank_scenario[
                "Target_Reached"
            ]
        ]
    )

    success_count = len(
        successful
    )

    probability_target = (
        success_count
        / NUMBER_OF_SIMULATIONS
    )

    ci_lower, ci_upper = (
        wilson_interval(
            success_count,
            NUMBER_OF_SIMULATIONS
        )
    )

    if success_count > 0:

        average_target_matches = (
            successful[
                "Matches_to_Target"
            ].mean()
        )

        median_target_matches = (
            successful[
                "Matches_to_Target"
            ].median()
        )

    else:

        average_target_matches = np.nan

        median_target_matches = np.nan

    if success_count >= 100:

        time_reliability = "Adequate"

    else:

        time_reliability = "Low sample"

    theoretical_stars = (
        MATCHES_PER_SEASON
        *
        (
            2 * probability
            - 1
        )
    )

    rank_scenario_rows.append({

        "True_Win_Rate":
            probability * 100,

        "Successful_Journeys":
            success_count,

        "P_Reach_50_Stars":
            probability_target * 100,

        "Target_CI_Lower":
            ci_lower * 100,

        "Target_CI_Upper":
            ci_upper * 100,

        "Average_Terminal_Stars":
            rank_scenario[
                "Terminal_Stars_After_Horizon"
            ].mean(),

        "Theoretical_Terminal_Stars":
            theoretical_stars,

        "Average_Matches_to_Target":
            average_target_matches,

        "Median_Matches_to_Target":
            median_target_matches,

        "Time_Estimate_Reliability":
            time_reliability,

        "Average_Maximum_Drawdown":
            rank_scenario[
                "Maximum_Drawdown_To_Stop"
            ].mean(),

        "Drawdown_95th_Percentile":
            np.percentile(
                rank_scenario[
                    "Maximum_Drawdown_To_Stop"
                ],
                95
            ),

        "P_10_Plus_Drawdown":
            (
                rank_scenario[
                    "Maximum_Drawdown_To_Stop"
                ] >= 10
            ).mean() * 100,

        "P_15_Plus_Drawdown":
            (
                rank_scenario[
                    "Maximum_Drawdown_To_Stop"
                ] >= 15
            ).mean() * 100,

        "P_20_Plus_Drawdown":
            (
                rank_scenario[
                    "Maximum_Drawdown_To_Stop"
                ] >= 20
            ).mean() * 100
    })


rank_scenario_summary = pd.DataFrame(
    rank_scenario_rows
)


print("\n========================================")
print("RANK SCENARIO ANALYSIS")
print("========================================")

print(
    rank_scenario_summary
    .round(2)
    .to_string(index=False)
)


horizon_rows = []

for horizon in HORIZONS:

    horizon_simulations = (
        simulate_rank_journeys(
            BASE_WIN_PROBABILITY,
            horizon,
            NUMBER_OF_SIMULATIONS,
            TARGET_STARS,
            rng
        )
    )

    horizon_successes = (
        horizon_simulations[
            horizon_simulations[
                "Target_Reached"
            ]
        ]
    )

    success_count = len(
        horizon_successes
    )

    probability_target = (
        success_count
        / NUMBER_OF_SIMULATIONS
    )

    ci_lower, ci_upper = (
        wilson_interval(
            success_count,
            NUMBER_OF_SIMULATIONS
        )
    )

    if success_count > 0:

        average_time = (
            horizon_successes[
                "Matches_to_Target"
            ].mean()
        )

    else:

        average_time = np.nan

    horizon_rows.append({

        "Match_Horizon":
            horizon,

        "Successful_Journeys":
            success_count,

        "P_Reach_50_Stars":
            probability_target * 100,

        "Target_CI_Lower":
            ci_lower * 100,

        "Target_CI_Upper":
            ci_upper * 100,

        "Average_Matches_to_Target":
            average_time,

        "Average_Terminal_Stars":
            horizon_simulations[
                "Terminal_Stars_After_Horizon"
            ].mean(),

        "Average_Maximum_Drawdown":
            horizon_simulations[
                "Maximum_Drawdown_To_Stop"
            ].mean()
    })


horizon_summary = pd.DataFrame(
    horizon_rows
)


print("\n========================================")
print("TIME-HORIZON SENSITIVITY")
print("========================================")

print(
    horizon_summary
    .round(2)
    .to_string(index=False)
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    match_scenario_summary[
        "True_Win_Rate"
    ],
    match_scenario_summary[
        "P_5_Plus_Loss_Streak"
    ],
    marker="o",
    label="5+ Loss Streak"
)

plt.plot(
    match_scenario_summary[
        "True_Win_Rate"
    ],
    match_scenario_summary[
        "P_7_Plus_Loss_Streak"
    ],
    marker="o",
    label="7+ Loss Streak"
)

plt.plot(
    match_scenario_summary[
        "True_Win_Rate"
    ],
    match_scenario_summary[
        "P_10_Plus_Loss_Streak"
    ],
    marker="o",
    label="10+ Loss Streak"
)

plt.xlabel(
    "True Win Probability (%)"
)

plt.ylabel(
    "Probability of Streak (%)"
)

plt.title(
    "Losing-Streak Risk Across Player Strength\n"
    "10,000 Simulated 200-Match Seasons"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/01_streak_risk.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(
    figsize=(10, 6)
)

plt.hist(
    baseline_match_simulations[
        "Observed_Win_Rate"
    ] * 100,
    bins=25
)

plt.axvline(
    BASE_WIN_PROBABILITY * 100,
    linestyle="--",
    label="True Win Probability"
)

plt.axvline(
    50,
    linestyle="--",
    label="50% Benchmark"
)

plt.xlabel(
    "Observed Win Rate (%)"
)

plt.ylabel(
    "Number of Simulated Seasons"
)

plt.title(
    "Observed Win-Rate Distribution\n"
    "55% True WR, 200 Matches"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/02_observed_wr_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

target_probabilities = (
    rank_scenario_summary[
        "P_Reach_50_Stars"
    ]
)

target_lower = (
    rank_scenario_summary[
        "Target_CI_Lower"
    ]
)

target_upper = (
    rank_scenario_summary[
        "Target_CI_Upper"
    ]
)

lower_errors = (
    target_probabilities
    - target_lower
)

upper_errors = (
    target_upper
    - target_probabilities
)

plt.figure(
    figsize=(10, 6)
)

plt.errorbar(
    rank_scenario_summary[
        "True_Win_Rate"
    ],
    target_probabilities,
    yerr=[
        lower_errors,
        upper_errors
    ],
    marker="o",
    capsize=4
)

plt.xlabel(
    "True Win Probability (%)"
)

plt.ylabel(
    "Probability of Reaching +50 Stars (%)"
)

plt.title(
    "Rank-Target Probability vs Player Strength\n"
    "95% Monte Carlo Confidence Intervals"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/03_rank_target_probability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    rank_scenario_summary[
        "True_Win_Rate"
    ],
    rank_scenario_summary[
        "Average_Terminal_Stars"
    ],
    marker="o",
    label="Simulated"
)

plt.plot(
    rank_scenario_summary[
        "True_Win_Rate"
    ],
    rank_scenario_summary[
        "Theoretical_Terminal_Stars"
    ],
    marker="o",
    label="Theoretical"
)

plt.xlabel(
    "True Win Probability (%)"
)

plt.ylabel(
    "Average Net Stars After 200 Matches"
)

plt.title(
    "Monte Carlo Validation\n"
    "Simulated vs Theoretical Rank Progression"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/04_rank_model_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    rank_scenario_summary[
        "True_Win_Rate"
    ],
    rank_scenario_summary[
        "P_10_Plus_Drawdown"
    ],
    marker="o",
    label="10+ Star Drawdown"
)

plt.plot(
    rank_scenario_summary[
        "True_Win_Rate"
    ],
    rank_scenario_summary[
        "P_15_Plus_Drawdown"
    ],
    marker="o",
    label="15+ Star Drawdown"
)

plt.plot(
    rank_scenario_summary[
        "True_Win_Rate"
    ],
    rank_scenario_summary[
        "P_20_Plus_Drawdown"
    ],
    marker="o",
    label="20+ Star Drawdown"
)

plt.xlabel(
    "True Win Probability (%)"
)

plt.ylabel(
    "Probability of Drawdown (%)"
)

plt.title(
    "Rank Drawdown Risk Across Player Strength"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/05_drawdown_risk.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    horizon_summary[
        "Match_Horizon"
    ],
    horizon_summary[
        "P_Reach_50_Stars"
    ],
    marker="o"
)

plt.xlabel(
    "Maximum Number of Matches"
)

plt.ylabel(
    "Probability of Reaching +50 Stars (%)"
)

plt.title(
    "Time-Horizon Sensitivity\n"
    "55% True Win Probability"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/06_horizon_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

baseline_summary = pd.DataFrame({

    "Metric": [

        "True Win Probability (%)",
        "Matches per Season",
        "Number of Simulations",

        "Average Observed Win Rate (%)",
        "Theoretical Observed-WR SE (%)",
        "Simulated Observed-WR SD (%)",

        "Observed WR 2.5th Percentile (%)",
        "Observed WR 97.5th Percentile (%)",

        "Probability Observed WR Below 50% (%)",

        "Average Longest Losing Streak",
        "95th Percentile Losing Streak",

        "Probability of 5+ Loss Streak (%)",
        "Probability of 7+ Loss Streak (%)",
        "Probability of 10+ Loss Streak (%)",

        "Rank Target (Net Stars)",

        "Successful Rank Journeys",

        "Probability of Reaching Target (%)",

        "Target Probability CI Lower (%)",
        "Target Probability CI Upper (%)",

        "Average Terminal Stars",

        "Theoretical Terminal Stars",

        "Simulated Terminal-Star SD",

        "Theoretical Terminal-Star SD",

        "Average Maximum Drawdown",

        "95th Percentile Maximum Drawdown",

        "Probability of 10+ Star Drawdown (%)",
        "Probability of 15+ Star Drawdown (%)",
        "Probability of 20+ Star Drawdown (%)",

        "Average Matches to Target",
        "Median Matches to Target"
    ],

    "Value": [

        BASE_WIN_PROBABILITY * 100,

        MATCHES_PER_SEASON,

        NUMBER_OF_SIMULATIONS,

        average_observed_wr * 100,

        theoretical_wr_standard_error * 100,

        observed_wr_std * 100,

        observed_wr_2_5 * 100,

        observed_wr_97_5 * 100,

        prob_wr_below_50 * 100,

        average_longest_loss,

        loss_streak_95th,

        prob_5_loss * 100,

        prob_7_loss * 100,

        prob_10_loss * 100,

        TARGET_STARS,

        successful_count,

        target_probability * 100,

        target_ci_lower * 100,

        target_ci_upper * 100,

        average_terminal_stars,

        theoretical_terminal_mean,

        terminal_star_std,

        theoretical_terminal_sd,

        average_drawdown,

        drawdown_95th,

        prob_10_drawdown * 100,

        prob_15_drawdown * 100,

        prob_20_drawdown * 100,

        average_matches_to_target,

        median_matches_to_target
    ]
})


print("\n========================================")
print("FINAL BASELINE SUMMARY")
print("========================================")

print(
    baseline_summary
    .round(2)
    .to_string(index=False)
)


assumptions = pd.DataFrame({

    "Assumption": [

        "Independent matches",

        "Constant win probability",

        "Win star movement",

        "Loss star movement",

        "Rank target",

        "Matchmaking mechanics",

        "Star protection and bonus mechanics"
    ],

    "Description": [

        "Each simulated match outcome is independent.",

        "Player true win probability stays constant during each simulation.",

        "Each win adds exactly 1 net star.",

        "Each loss removes exactly 1 net star.",

        "Target is defined as +50 net stars from the starting position.",

        "Changes in matchmaking difficulty are not modelled.",

        "Star protection, bonus stars and similar MLBB mechanics are excluded."
    ]
})


baseline_match_simulations.to_excel(
    f"{OUTPUT_FOLDER}/baseline_match_simulations.xlsx",
    index=False
)

match_scenario_summary.to_excel(
    f"{OUTPUT_FOLDER}/match_scenario_analysis.xlsx",
    index=False
)

baseline_rank_simulations.to_excel(
    f"{OUTPUT_FOLDER}/baseline_rank_simulations.xlsx",
    index=False
)

rank_scenario_summary.to_excel(
    f"{OUTPUT_FOLDER}/rank_scenario_analysis.xlsx",
    index=False
)

horizon_summary.to_excel(
    f"{OUTPUT_FOLDER}/horizon_sensitivity.xlsx",
    index=False
)

baseline_summary.to_excel(
    f"{OUTPUT_FOLDER}/baseline_summary.xlsx",
    index=False
)


with pd.ExcelWriter(
    f"{OUTPUT_FOLDER}/MLBB_Monte_Carlo_Analysis.xlsx",
    engine="openpyxl"
) as writer:

    baseline_summary.to_excel(
        writer,
        sheet_name="Baseline Summary",
        index=False
    )

    match_scenario_summary.to_excel(
        writer,
        sheet_name="Match Scenarios",
        index=False
    )

    rank_scenario_summary.to_excel(
        writer,
        sheet_name="Rank Scenarios",
        index=False
    )

    horizon_summary.to_excel(
        writer,
        sheet_name="Horizon Sensitivity",
        index=False
    )

    assumptions.to_excel(
        writer,
        sheet_name="Model Assumptions",
        index=False
    )

    baseline_match_simulations.to_excel(
        writer,
        sheet_name="10000 Match Sims",
        index=False
    )

    baseline_rank_simulations.to_excel(
        writer,
        sheet_name="10000 Rank Sims",
        index=False
    )


print("\n========================================")
print("PROJECT #2 COMPLETE")
print("========================================")

print(
    "All results have been saved inside:",
    OUTPUT_FOLDER
)
