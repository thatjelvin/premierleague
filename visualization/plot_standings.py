"""Visualization module for Premier League predictions."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_position_probability_heatmap(team_probabilities: dict,
                                       output_path: str = None):
    """Create heatmap of position probabilities for each team."""
    if output_path is None:
        output_path = f"{config.PREDICTIONS_DIR}/position_probabilities.png"

    # Prepare data for heatmap
    teams = []
    positions = list(range(1, 21))  # 1-20 positions
    data = []

    for team_id, probs in sorted(team_probabilities.items(),
                                   key=lambda x: x[1]['expected_final_points'],
                                   reverse=True):
        teams.append(probs['team_name'])
        row = []
        for pos in positions:
            prob = probs['position_probabilities'].get(pos, 0)
            row.append(prob * 100)  # Convert to percentage
        data.append(row)

    df = pd.DataFrame(data, columns=positions, index=teams)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Custom colormap (white to dark green)
    colors = ['#ffffff', '#e6f3e6', '#c6e6c6', '#8fd98f',
              '#5cb85c', '#3d8b3d', '#2d6a2d', '#1a4a1a']
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.1f', cmap=cmap,
                cbar_kws={'label': 'Probability (%)', 'shrink': 0.8},
                linewidths=0.5, ax=ax, vmin=0, vmax=50)

    # Styling
    ax.set_xlabel('Final Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Team', fontsize=12, fontweight='bold')
    ax.set_title('Probability of Each Team Finishing in Each Position\n(2024/25 Premier League)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add vertical lines for key positions
    ax.axvline(x=4, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=6, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=17, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.5, label='Top 4 (Champions League)'),
        mpatches.Patch(color='orange', alpha=0.5, label='Top 6 (Europe)'),
        mpatches.Patch(color='black', alpha=0.5, label='Relegation Zone')
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.15, 1), title='Key Positions')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Position probability heatmap saved to {output_path}")
    plt.close()


def plot_final_standings(final_standings: pd.DataFrame,
                         output_path: str = None):
    """Create bar chart of predicted final standings with confidence intervals."""
    if output_path is None:
        output_path = f"{config.PREDICTIONS_DIR}/final_standings_chart.png"

    fig, ax = plt.subplots(figsize=(12, 10))

    teams = final_standings['team_name'].tolist()
    y_pos = np.arange(len(teams))

    # Points data
    expected = final_standings['expected_final_points'].values
    current = final_standings['current_points'].values
    std = final_standings['points_std'].values

    # Create horizontal bar chart
    bars = ax.barh(y_pos, expected, xerr=std, capsize=3,
                   color=['#1f77b4' if i < 4 else '#ff7f0e' if i < 6
                          else '#d62728' if i >= 17 else '#7f7f7f'
                          for i in range(len(teams))],
                   alpha=0.8, error_kw={'linewidth': 1.5, 'ecolor': 'black'})

    # Add current points as markers
    ax.scatter(current, y_pos, color='white', s=50, zorder=5,
               edgecolors='black', linewidth=1.5, marker='o', label='Current Points')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i+1}. {team}" for i, team in enumerate(teams)])
    ax.invert_yaxis()
    ax.set_xlabel('Points', fontsize=12, fontweight='bold')
    ax.set_title('Predicted Final Standings 2024/25\n(with 1σ confidence intervals)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label='Top 4 (Champions League)'),
        mpatches.Patch(color='#ff7f0e', label='Top 6 (Europe)'),
        mpatches.Patch(color='#d62728', label='Relegation Zone'),
        mpatches.Patch(color='#7f7f7f', label='Mid Table'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=8, label='Current Points')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add point values on bars
    for i, (bar, exp) in enumerate(zip(bars, expected)):
        ax.text(exp + 2, i, f'{exp:.0f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Final standings chart saved to {output_path}")
    plt.close()


def plot_outcome_probabilities(final_standings: pd.DataFrame,
                                output_path: str = None):
    """Create stacked bar chart of key outcome probabilities."""
    if output_path is None:
        output_path = f"{config.PREDICTIONS_DIR}/outcome_probabilities.png"

    fig, ax = plt.subplots(figsize=(14, 8))

    teams = final_standings['team_name'].tolist()
    y_pos = np.arange(len(teams))

    # Get probabilities
    champ = final_standings['champion_probability'].values
    top4 = final_standings['top4_probability'].values - champ  # Remaining
    top6 = final_standings['top6_probability'].values - final_standings['top4_probability'].values
    mid = 100 - final_standings['top6_probability'].values - final_standings['relegation_probability'].values
    releg = final_standings['relegation_probability'].values

    # Create stacked bars
    width = 0.7

    p1 = ax.barh(y_pos, champ, width, label='Champions', color='#ffd700')
    p2 = ax.barh(y_pos, top4, width, left=champ, label='Top 4', color='#4169e1')
    p3 = ax.barh(y_pos, top6, width, left=champ + top4, label='Top 6', color='#87ceeb')
    p4 = ax.barh(y_pos, mid, width, left=champ + top4 + top6, label='Mid Table', color='#808080')
    p5 = ax.barh(y_pos, releg, width, left=champ + top4 + top6 + mid, label='Relegation', color='#dc143c')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(teams)
    ax.invert_yaxis()
    ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Probability of Key Season Outcomes by Team',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', title='Outcome')
    ax.set_xlim(0, 100)

    # Add vertical reference lines
    ax.axvline(x=50, color='black', linestyle=':', alpha=0.3)
    ax.axvline(x=75, color='black', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Outcome probabilities chart saved to {output_path}")
    plt.close()


def plot_fixture_difficulty(fixtures_df: pd.DataFrame,
                             predictions_df: pd.DataFrame,
                             standings_df: pd.DataFrame,
                             output_path: str = None):
    """Plot remaining fixture difficulty for each team."""
    if output_path is None:
        output_path = f"{config.PREDICTIONS_DIR}/fixture_difficulty.png"

    # Calculate difficulty for each team's fixtures
    team_difficulties = {}

    for _, team in standings_df.iterrows():
        team_id = team['team_id']
        team_name = team['short_name']

        # Get team fixtures
        team_fixtures = predictions_df[
            (predictions_df['home_team_id'] == team_id) |
            (predictions_df['away_team_id'] == team_id)
        ].copy()

        difficulties = []
        for _, fixture in team_fixtures.iterrows():
            if fixture['home_team_id'] == team_id:
                # Playing at home
                difficulty = 1 - fixture['prob_home_win'] - 0.3 * fixture['prob_draw']
            else:
                # Playing away
                difficulty = 1 - fixture['prob_away_win'] - 0.3 * fixture['prob_draw']
            difficulties.append(difficulty)

        team_difficulties[team_name] = difficulties

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data
    max_fixtures = max(len(v) for v in team_difficulties.values())
    teams = sorted(team_difficulties.keys(),
                   key=lambda x: np.mean(team_difficulties[x]), reverse=True)

    data = []
    for team in teams:
        row = team_difficulties[team]
        # Pad if needed
        while len(row) < max_fixtures:
            row.append(np.nan)
        data.append(row)

    df = pd.DataFrame(data, index=teams)

    # Create heatmap
    cmap = sns.diverging_palette(10, 133, s=85, l=55, n=100, as_cmap=True)
    sns.heatmap(df, annot=False, cmap=cmap, cbar_kws={'label': 'Difficulty'},
                linewidths=0.5, ax=ax, vmin=0, vmax=1)

    # Styling
    ax.set_xlabel('Remaining Fixtures (ordered by date)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Team', fontsize=12, fontweight='bold')
    ax.set_title('Remaining Fixture Difficulty\n(Darker = Harder, Lighter = Easier)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Fixture difficulty heatmap saved to {output_path}")
    plt.close()


def plot_points_distribution(results: dict,
                              output_path: str = None):
    """Plot distribution of final points for top teams."""
    if output_path is None:
        output_path = f"{config.PREDICTIONS_DIR}/points_distribution.png"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    team_probs = results['team_probabilities']

    # Get top 6 teams by expected points
    top_teams = sorted(team_probs.items(),
                       key=lambda x: x[1]['expected_final_points'],
                       reverse=True)[:6]

    for idx, (team_id, team_data) in enumerate(top_teams):
        ax = axes[idx]

        points = results['points_distribution'][team_id]
        team_name = team_data['team_name']

        # Create histogram
        ax.hist(points, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(team_data['expected_final_points'], color='red', linestyle='--',
                   linewidth=2, label=f'Expected: {team_data["expected_final_points"]:.1f}')
        ax.axvline(team_data['current_points'], color='green', linestyle=':',
                   linewidth=2, label=f'Current: {team_data["current_points"]}')

        ax.set_xlabel('Final Points', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{team_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Distribution of Simulated Final Points (Top 6 Teams)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Points distribution chart saved to {output_path}")
    plt.close()


def create_all_visualizations():
    """Generate all visualization outputs."""
    print("\nGenerating visualizations...")

    # Load data
    final_standings = pd.read_csv(f"{config.PREDICTIONS_DIR}/final_standings.csv")
    predictions = pd.read_csv(f"{config.PREDICTIONS_DIR}/match_predictions.csv")
    fixtures = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv")
    standings = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv")

    # Import simulation results
    from simulation.season_simulator import SeasonSimulator
    simulator = SeasonSimulator(standings, fixtures, predictions)
    _, results = simulator.get_final_standings_prediction()

    # Generate plots
    plot_final_standings(final_standings)
    plot_position_probability_heatmap(results['team_probabilities'])
    plot_outcome_probabilities(final_standings)
    plot_fixture_difficulty(fixtures, predictions, standings)
    plot_points_distribution(results)

    print("\nAll visualizations generated successfully!")


def main():
    """Generate all visualizations."""
    create_all_visualizations()


if __name__ == '__main__':
    main()
