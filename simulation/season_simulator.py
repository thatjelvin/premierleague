"""Monte Carlo season simulation for Premier League predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MatchSimulator:
    """Simulate individual match outcomes based on predicted probabilities."""

    def __init__(self, random_state: int = None):
        self.rng = np.random.RandomState(random_state)

    def simulate_match(self, home_win_prob: float, draw_prob: float,
                       away_win_prob: float, expected_home_goals: float,
                       expected_away_goals: float) -> Dict:
        """Simulate a single match and return the outcome."""
        # Normalize probabilities
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total

        # Sample outcome
        outcome = self.rng.choice(
            [0, 1, 2],  # 0=Home Win, 1=Draw, 2=Away Win
            p=[home_win_prob, draw_prob, away_win_prob]
        )

        # Sample goals using Poisson distribution with some variation
        if outcome == 0:  # Home win
            home_goals = max(1, int(self.rng.poisson(expected_home_goals * 1.3)))
            away_goals = self.rng.poisson(expected_away_goals * 0.7)
            away_goals = min(away_goals, home_goals - 1)
        elif outcome == 1:  # Draw
            goals = self.rng.poisson((expected_home_goals + expected_away_goals) / 2)
            home_goals = away_goals = goals
        else:  # Away win
            away_goals = max(1, int(self.rng.poisson(expected_away_goals * 1.3)))
            home_goals = self.rng.poisson(expected_home_goals * 0.7)
            home_goals = min(home_goals, away_goals - 1)

        # Determine points
        if home_goals > away_goals:
            home_points, away_points = 3, 0
            winner = 'HOME'
        elif home_goals == away_goals:
            home_points, away_points = 1, 1
            winner = 'DRAW'
        else:
            home_points, away_points = 0, 3
            winner = 'AWAY'

        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_points': home_points,
            'away_points': away_points,
            'winner': winner
        }

    def simulate_match_simple(self, home_win_prob: float, draw_prob: float,
                               away_win_prob: float) -> Tuple[int, int]:
        """Simple simulation returning just points."""
        total = home_win_prob + draw_prob + away_win_prob
        probs = [home_win_prob/total, draw_prob/total, away_win_prob/total]

        outcome = self.rng.choice([0, 1, 2], p=probs)

        if outcome == 0:
            return 3, 0  # Home win
        elif outcome == 1:
            return 1, 1  # Draw
        else:
            return 0, 3  # Away win


class SeasonSimulator:
    """Monte Carlo simulation of remaining season fixtures."""

    def __init__(self, current_standings: pd.DataFrame,
                 remaining_fixtures: pd.DataFrame,
                 match_predictions: pd.DataFrame,
                 n_simulations: int = config.N_SIMULATIONS):
        self.current_standings = current_standings.copy()
        self.remaining_fixtures = remaining_fixtures.copy()
        self.match_predictions = match_predictions.copy()
        self.n_simulations = n_simulations

        # Create lookup for predictions
        self.pred_lookup = {}
        for _, pred in match_predictions.iterrows():
            key = (pred['home_team_id'], pred['away_team_id'])
            self.pred_lookup[key] = pred

        # Initialize team stats
        self.team_stats = {}
        for _, team in current_standings.iterrows():
            self.team_stats[team['team_id']] = {
                'team_name': team['team_name'],
                'short_name': team['short_name'],
                'played': team['played'],
                'won': team['won'],
                'drawn': team['drawn'],
                'lost': team['lost'],
                'goals_for': team['goals_for'],
                'goals_against': team['goals_against'],
                'points': team['points']
            }

    def _get_prediction(self, home_id: int, away_id: int) -> Dict:
        """Get prediction for a fixture."""
        key = (home_id, away_id)

        if key in self.pred_lookup:
            pred = self.pred_lookup[key]
            return {
                'home_win': pred['prob_home_win'],
                'draw': pred['prob_draw'],
                'away_win': pred['prob_away_win'],
                'expected_home_goals': pred['expected_home_goals'],
                'expected_away_goals': pred['expected_away_goals']
            }

        # Fallback if prediction not found
        return {
            'home_win': 0.45,
            'draw': 0.25,
            'away_win': 0.30,
            'expected_home_goals': 1.4,
            'expected_away_goals': 1.1
        }

    def _simulate_season_once(self, seed: int) -> pd.DataFrame:
        """Simulate the remaining season once."""
        simulator = MatchSimulator(random_state=seed)

        # Copy current stats
        season_stats = {
            tid: stats.copy()
            for tid, stats in self.team_stats.items()
        }

        # Simulate each remaining fixture
        for _, fixture in self.remaining_fixtures.iterrows():
            home_id = fixture['home_team_id']
            away_id = fixture['away_team_id']

            pred = self._get_prediction(home_id, away_id)

            result = simulator.simulate_match(
                pred['home_win'],
                pred['draw'],
                pred['away_win'],
                pred['expected_home_goals'],
                pred['expected_away_goals']
            )

            # Update stats
            season_stats[home_id]['played'] += 1
            season_stats[home_id]['goals_for'] += result['home_goals']
            season_stats[home_id]['goals_against'] += result['away_goals']
            season_stats[home_id]['points'] += result['home_points']

            season_stats[away_id]['played'] += 1
            season_stats[away_id]['goals_for'] += result['away_goals']
            season_stats[away_id]['goals_against'] += result['home_goals']
            season_stats[away_id]['points'] += result['away_points']

            # Track wins/losses/draws
            if result['winner'] == 'HOME':
                season_stats[home_id]['won'] += 1
                season_stats[away_id]['lost'] += 1
            elif result['winner'] == 'AWAY':
                season_stats[home_id]['lost'] += 1
                season_stats[away_id]['won'] += 1
            else:
                season_stats[home_id]['drawn'] += 1
                season_stats[away_id]['drawn'] += 1

        # Create final table
        table = []
        for team_id, stats in season_stats.items():
            stats['team_id'] = team_id
            stats['goal_difference'] = stats['goals_for'] - stats['goals_against']
            table.append(stats)

        df = pd.DataFrame(table)
        df = df.sort_values(
            ['points', 'goal_difference', 'goals_for'],
            ascending=[False, False, False]
        )
        df['position'] = range(1, len(df) + 1)

        return df

    def run_simulation(self) -> Dict:
        """Run Monte Carlo simulation of remaining season."""
        print(f"Running {self.n_simulations} season simulations...")

        all_results = []
        position_counts = defaultdict(lambda: defaultdict(int))
        points_distribution = defaultdict(list)

        # Run simulations with progress bar
        for i in tqdm(range(self.n_simulations), desc="Simulating"):
            result = self._simulate_season_once(seed=i)
            all_results.append(result)

            # Track positions
            for _, team in result.iterrows():
                position_counts[team['team_id']][team['position']] += 1
                points_distribution[team['team_id']].append(team['points'])

        # Calculate statistics
        team_probabilities = {}

        for team_id in self.team_stats.keys():
            team_name = self.team_stats[team_id]['team_name']

            positions = position_counts[team_id]
            total = sum(positions.values())

            # Position probabilities
            position_probs = {
                pos: count / total
                for pos, count in positions.items()
            }

            # Specific outcome probabilities
            champ_prob = sum(p for pos, p in position_probs.items() if pos == 1)
            top4_prob = sum(p for pos, p in position_probs.items() if pos <= 4)
            top6_prob = sum(p for pos, p in position_probs.items() if pos <= 6)
            relegation_prob = sum(p for pos, p in position_probs.items() if pos >= 18)

            # Points statistics
            points = points_distribution[team_id]
            expected_points = np.mean(points)
            points_std = np.std(points)

            team_probabilities[team_id] = {
                'team_name': team_name,
                'current_points': self.team_stats[team_id]['points'],
                'expected_points': expected_points,
                'points_std': points_std,
                'expected_final_points': expected_points,
                'champion_probability': champ_prob,
                'top4_probability': top4_prob,
                'top6_probability': top6_prob,
                'relegation_probability': relegation_prob,
                'position_probabilities': position_probs,
                'most_likely_position': max(position_probs.items(), key=lambda x: x[1])[0]
                if position_probs else 10
            }

        return {
            'team_probabilities': team_probabilities,
            'all_results': all_results,
            'position_counts': dict(position_counts),
            'points_distribution': {k: v for k, v in points_distribution.items()}
        }

    def get_final_standings_prediction(self) -> pd.DataFrame:
        """Get predicted final standings."""
        results = self.run_simulation()

        # Create summary table
        summary = []
        for team_id, probs in results['team_probabilities'].items():
            summary.append({
                'team_id': team_id,
                'team_name': probs['team_name'],
                'current_points': probs['current_points'],
                'expected_final_points': round(probs['expected_final_points'], 1),
                'expected_final_position': probs['most_likely_position'],
                'champion_probability': round(probs['champion_probability'] * 100, 1),
                'top4_probability': round(probs['top4_probability'] * 100, 1),
                'top6_probability': round(probs['top6_probability'] * 100, 1),
                'relegation_probability': round(probs['relegation_probability'] * 100, 1),
                'points_std': round(probs['points_std'], 1)
            })

        df = pd.DataFrame(summary)
        df = df.sort_values('expected_final_points', ascending=False)
        df['predicted_position'] = range(1, len(df) + 1)

        return df, results


def main():
    """Run season simulation."""
    # Load data
    standings = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv")
    fixtures = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv")
    predictions = pd.read_csv(f"{config.PREDICTIONS_DIR}/match_predictions.csv")

    print(f"Current standings: {len(standings)} teams")
    print(f"Remaining fixtures: {len(fixtures)} matches")

    # Run simulation
    simulator = SeasonSimulator(standings, fixtures, predictions,
                                   n_simulations=config.N_SIMULATIONS)
    final_standings, all_results = simulator.get_final_standings_prediction()

    # Save results
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    final_standings.to_csv(f"{config.PREDICTIONS_DIR}/final_standings.csv", index=False)

    # Print results
    print("\n" + "="*100)
    print("PREDICTED FINAL STANDINGS 2024/25")
    print("="*100)
    print(f"{'Pos':<4} {'Team':<25} {'Current':<8} {'Expected':<9} {'Champ%':<8} {'Top4%':<8} {'Top6%':<8} {'Rel%':<8}")
    print("-"*100)

    for _, team in final_standings.iterrows():
        print(f"{team['predicted_position']:<4} {team['team_name']:<25} "
              f"{team['current_points']:<8} {team['expected_final_points']:<9} "
              f"{team['champion_probability']:<8} {team['top4_probability']:<8} "
              f"{team['top6_probability']:<8} {team['relegation_probability']:<8}")

    print("\nKey Predictions:")
    print("-"*100)

    # Champion
    champ_candidates = final_standings.nlargest(3, 'champion_probability')
    print("\nTitle Race:")
    for _, team in champ_candidates.iterrows():
        if team['champion_probability'] > 0:
            print(f"  {team['team_name']}: {team['champion_probability']:.1f}% chance")

    # Top 4
    top4_candidates = final_standings.nlargest(6, 'top4_probability')
    print("\nTop 4 Race:")
    for _, team in top4_candidates.iterrows():
        if team['top4_probability'] > 5:
            print(f"  {team['team_name']}: {team['top4_probability']:.1f}% chance")

    # Relegation
    rel_candidates = final_standings.nlargest(5, 'relegation_probability')
    print("\nRelegation Battle:")
    for _, team in rel_candidates.iterrows():
        if team['relegation_probability'] > 5:
            print(f"  {team['team_name']}: {team['relegation_probability']:.1f}% chance")

    print(f"\n\nResults saved to {config.PREDICTIONS_DIR}/final_standings.csv")

    return final_standings, all_results


if __name__ == '__main__':
    main()
