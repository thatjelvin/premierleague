"""Feature engineering module for Premier League prediction model."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeatureBuilder:
    """Build features for match prediction from historical data."""

    def __init__(self):
        self.team_stats_cache = {}
        self.head_to_head_cache = {}

    def create_match_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable: match outcome (H=0, D=1, A=2)."""
        df = df.copy()
        df['outcome'] = df.apply(lambda row:
            0 if row['home_goals'] > row['away_goals']  # Home win
            else 1 if row['home_goals'] == row['away_goals']  # Draw
            else 2, axis=1)  # Away win
        return df

    def calculate_team_form(self, df: pd.DataFrame, team_id: int,
                           match_date: str, window: int = 5) -> Dict:
        """Calculate team form (points from last N matches)."""
        # Get matches before the given date
        team_matches = df[
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['date'] < match_date)
        ].sort_values('date', ascending=False).head(window)

        if len(team_matches) == 0:
            return {'points': 0, 'goals_scored': 0, 'goals_conceded': 0,
                    'wins': 0, 'draws': 0, 'losses': 0}

        points = 0
        goals_scored = 0
        goals_conceded = 0
        wins = 0
        draws = 0
        losses = 0

        for _, match in team_matches.iterrows():
            if match['home_team_id'] == team_id:
                goals_scored += match['home_goals']
                goals_conceded += match['away_goals']
                if match['home_goals'] > match['away_goals']:
                    points += 3
                    wins += 1
                elif match['home_goals'] == match['away_goals']:
                    points += 1
                    draws += 1
                else:
                    losses += 1
            else:
                goals_scored += match['away_goals']
                goals_conceded += match['home_goals']
                if match['away_goals'] > match['home_goals']:
                    points += 3
                    wins += 1
                elif match['away_goals'] == match['home_goals']:
                    points += 1
                    draws += 1
                else:
                    losses += 1

        return {
            'points': points / max(len(team_matches), 1) * (5 / window),  # Normalize to 5-game equivalent
            'goals_scored': goals_scored / max(len(team_matches), 1),
            'goals_conceded': goals_conceded / max(len(team_matches), 1),
            'wins': wins,
            'draws': draws,
            'losses': losses
        }

    def calculate_home_away_form(self, df: pd.DataFrame, team_id: int,
                                   match_date: str, is_home: bool,
                                   window: int = 5) -> Dict:
        """Calculate home or away specific form."""
        if is_home:
            team_matches = df[
                (df['home_team_id'] == team_id) &
                (df['date'] < match_date)
            ].sort_values('date', ascending=False).head(window)
        else:
            team_matches = df[
                (df['away_team_id'] == team_id) &
                (df['date'] < match_date)
            ].sort_values('date', ascending=False).head(window)

        if len(team_matches) == 0:
            return {'points': 0, 'goals_scored': 0, 'goals_conceded': 0}

        points = 0
        goals_scored = 0
        goals_conceded = 0

        for _, match in team_matches.iterrows():
            if is_home:
                goals_scored += match['home_goals']
                goals_conceded += match['away_goals']
                if match['home_goals'] > match['away_goals']:
                    points += 3
                elif match['home_goals'] == match['away_goals']:
                    points += 1
            else:
                goals_scored += match['away_goals']
                goals_conceded += match['home_goals']
                if match['away_goals'] > match['home_goals']:
                    points += 3
                elif match['away_goals'] == match['home_goals']:
                    points += 1

        return {
            'points': points / max(len(team_matches), 1) * (5 / window),
            'goals_scored': goals_scored / max(len(team_matches), 1),
            'goals_conceded': goals_conceded / max(len(team_matches), 1)
        }

    def calculate_season_stats(self, df: pd.DataFrame, team_id: int,
                                match_date: str) -> Dict:
        """Calculate season-level statistics before a given date."""
        home_matches = df[
            (df['home_team_id'] == team_id) &
            (df['date'] < match_date)
        ]
        away_matches = df[
            (df['away_team_id'] == team_id) &
            (df['date'] < match_date)
        ]

        total_matches = len(home_matches) + len(away_matches)

        if total_matches == 0:
            return {
                'season_ppg': 1.0,
                'home_ppg': 1.0,
                'away_ppg': 1.0,
                'season_gpg': 1.4,
                'season_gcpg': 1.4
            }

        # Calculate points per game
        home_points = sum([
            3 if r['home_goals'] > r['away_goals'] else
            1 if r['home_goals'] == r['away_goals'] else 0
            for _, r in home_matches.iterrows()
        ])
        away_points = sum([
            3 if r['away_goals'] > r['home_goals'] else
            1 if r['away_goals'] == r['home_goals'] else 0
            for _, r in away_matches.iterrows()
        ])

        home_ppg = home_points / max(len(home_matches), 1)
        away_ppg = away_points / max(len(away_matches), 1)
        total_ppg = (home_points + away_points) / total_matches

        # Goals per game
        home_goals = home_matches['home_goals'].sum() if len(home_matches) > 0 else 0
        away_goals = away_matches['away_goals'].sum() if len(away_matches) > 0 else 0
        goals_scored = home_goals + away_goals

        home_conceded = home_matches['away_goals'].sum() if len(home_matches) > 0 else 0
        away_conceded = away_matches['home_goals'].sum() if len(away_matches) > 0 else 0
        goals_conceded = home_conceded + away_conceded

        return {
            'season_ppg': total_ppg,
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'season_gpg': goals_scored / total_matches,
            'season_gcpg': goals_conceded / total_matches
        }

    def get_head_to_head(self, df: pd.DataFrame, home_id: int,
                         away_id: int, match_date: str,
                         window: int = 5) -> Dict:
        """Get head-to-head history between two teams."""
        h2h = df[
            (((df['home_team_id'] == home_id) & (df['away_team_id'] == away_id)) |
             ((df['home_team_id'] == away_id) & (df['away_team_id'] == home_id))) &
            (df['date'] < match_date)
        ].sort_values('date', ascending=False).head(window)

        if len(h2h) == 0:
            return {'h2h_home_wins': 0, 'h2h_draws': 0, 'h2h_away_wins': 0}

        home_wins = 0
        draws = 0
        away_wins = 0

        for _, match in h2h.iterrows():
            if match['home_team_id'] == home_id:
                if match['home_goals'] > match['away_goals']:
                    home_wins += 1
                elif match['home_goals'] == match['away_goals']:
                    draws += 1
                else:
                    away_wins += 1
            else:  # The teams were reversed
                if match['away_goals'] > match['home_goals']:
                    home_wins += 1
                elif match['away_goals'] == match['home_goals']:
                    draws += 1
                else:
                    away_wins += 1

        total = len(h2h)
        return {
            'h2h_home_wins': home_wins / total,
            'h2h_draws': draws / total,
            'h2h_away_wins': away_wins / total
        }

    def calculate_league_position(self, df: pd.DataFrame, team_id: int,
                                   match_date: str, standings_df: pd.DataFrame = None) -> Dict:
        """Calculate current league position before a given match."""
        # Build a mini-table up to this point
        matches_before = df[df['date'] < match_date]

        team_points = {}
        team_gd = {}

        for _, match in matches_before.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            if home_id not in team_points:
                team_points[home_id] = 0
                team_gd[home_id] = 0
            if away_id not in team_points:
                team_points[away_id] = 0
                team_gd[away_id] = 0

            if match['home_goals'] > match['away_goals']:
                team_points[home_id] += 3
            elif match['home_goals'] == match['away_goals']:
                team_points[home_id] += 1
                team_points[away_id] += 1
            else:
                team_points[away_id] += 3

            team_gd[home_id] += match['home_goals'] - match['away_goals']
            team_gd[away_id] += match['away_goals'] - match['home_goals']

        # Create ranking
        if len(team_points) == 0:
            return {'position': 10, 'points': 0, 'games_played': 0}

        table = pd.DataFrame([
            {'team_id': tid, 'points': p, 'gd': team_gd.get(tid, 0)}
            for tid, p in team_points.items()
        ])

        table = table.sort_values(['points', 'gd'], ascending=[False, False])
        table['position'] = range(1, len(table) + 1)

        team_row = table[table['team_id'] == team_id]
        if len(team_row) == 0:
            return {'position': len(table) + 1, 'points': 0, 'games_played': 0}

        return {
            'position': team_row.iloc[0]['position'],
            'points': team_row.iloc[0]['points'],
            'games_played': len(matches_before[
                (matches_before['home_team_id'] == team_id) |
                (matches_before['away_team_id'] == team_id)
            ])
        }

    def calculate_days_since_last_match(self, df: pd.DataFrame, team_id: int,
                                        match_date: str) -> int:
        """Calculate days since team's last match."""
        team_matches = df[
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['date'] < match_date)
        ].sort_values('date', ascending=False)

        if len(team_matches) == 0:
            return 7  # Default to a week

        last_match_date = pd.to_datetime(team_matches.iloc[0]['date'])
        current_date = pd.to_datetime(match_date)

        return (current_date - last_match_date).days

    def build_features_for_match(self, df: pd.DataFrame, match: pd.Series,
                                  standings: Optional[pd.DataFrame] = None) -> Dict:
        """Build complete feature set for a single match."""
        home_id = match['home_team_id']
        away_id = match['away_team_id']
        match_date = match['date']

        # Current form (last 5 games)
        home_form = self.calculate_team_form(df, home_id, match_date, window=5)
        away_form = self.calculate_team_form(df, away_id, match_date, window=5)

        # Longer form (last 10 games)
        home_form_long = self.calculate_team_form(df, home_id, match_date, window=10)
        away_form_long = self.calculate_team_form(df, away_id, match_date, window=10)

        # Home/away specific form
        home_home_form = self.calculate_home_away_form(df, home_id, match_date, is_home=True)
        away_away_form = self.calculate_home_away_form(df, away_id, match_date, is_home=False)

        # Season stats
        home_season = self.calculate_season_stats(df, home_id, match_date)
        away_season = self.calculate_season_stats(df, away_id, match_date)

        # Head-to-head
        h2h = self.get_head_to_head(df, home_id, away_id, match_date)

        # League position
        home_league = self.calculate_league_position(df, home_id, match_date, standings)
        away_league = self.calculate_league_position(df, away_id, match_date, standings)

        # Rest days
        home_rest = self.calculate_days_since_last_match(df, home_id, match_date)
        away_rest = self.calculate_days_since_last_match(df, away_id, match_date)

        # Match context
        features = {
            # Form features
            'home_form_points': home_form['points'],
            'home_form_gf': home_form['goals_scored'],
            'home_form_ga': home_form['goals_conceded'],
            'home_form_wins': home_form['wins'],

            'away_form_points': away_form['points'],
            'away_form_gf': away_form['goals_scored'],
            'away_form_ga': away_form['goals_conceded'],
            'away_form_wins': away_form['wins'],

            # Long form
            'home_form_long': home_form_long['points'],
            'away_form_long': away_form_long['points'],

            # Home/Away specific
            'home_home_points': home_home_form['points'],
            'home_home_gf': home_home_form['goals_scored'],
            'home_home_ga': home_home_form['goals_conceded'],

            'away_away_points': away_away_form['points'],
            'away_away_gf': away_away_form['goals_scored'],
            'away_away_ga': away_away_form['goals_conceded'],

            # Season stats
            'home_season_ppg': home_season['season_ppg'],
            'home_season_gpg': home_season['season_gpg'],
            'home_season_gcpg': home_season['season_gcpg'],
            'home_home_ppg': home_season['home_ppg'],

            'away_season_ppg': away_season['season_ppg'],
            'away_season_gpg': away_season['season_gpg'],
            'away_season_gcpg': away_season['season_gcpg'],
            'away_away_ppg': away_season['away_ppg'],

            # Head-to-head
            'h2h_home_win_pct': h2h['h2h_home_wins'],
            'h2h_draw_pct': h2h['h2h_draws'],

            # League position
            'home_position': home_league['position'],
            'away_position': away_league['position'],
            'position_diff': away_league['position'] - home_league['position'],
            'points_diff': home_league['points'] - away_league['points'],

            # Rest
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'rest_diff': home_rest - away_rest,

            # Match context
            'matchday': match['matchday'],

            # IDs for reference
            'home_team_id': home_id,
            'away_team_id': away_id,
        }

        return features

    def build_training_features(self, matches_df: pd.DataFrame,
                                 standings_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build features for all matches in training data."""
        matches_df = matches_df.copy()
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        matches_df = matches_df.sort_values('date')

        # Add target
        matches_df = self.create_match_outcome(matches_df)

        features_list = []
        total = len(matches_df)

        print(f"Building features for {total} matches...")

        for idx, (_, match) in enumerate(matches_df.iterrows()):
            if idx % 500 == 0:
                print(f"  Processed {idx}/{total} matches...")

            # Only use matches from current season for form calculation
            season_matches = matches_df[matches_df['season'] == match['season']]

            features = self.build_features_for_match(season_matches, match, standings_df)
            features['outcome'] = match['outcome']
            features['match_id'] = match['match_id']
            features['season'] = match['season']

            features_list.append(features)

        return pd.DataFrame(features_list)

    def build_prediction_features(self, fixtures_df: pd.DataFrame,
                                   historical_df: pd.DataFrame,
                                   current_standings: pd.DataFrame) -> pd.DataFrame:
        """Build features for future matches to predict."""
        features_list = []

        # Combine historical data with current season for form calculation
        current_season_matches = historical_df[
            historical_df['season'] == config.CURRENT_SEASON
        ]

        for _, fixture in fixtures_df.iterrows():
            features = self.build_features_for_match(
                current_season_matches, fixture, current_standings
            )
            features['match_id'] = fixture['match_id']
            features['home_team'] = fixture['home_team']
            features['away_team'] = fixture['away_team']
            features['matchday'] = fixture['matchday']

            features_list.append(features)

        return pd.DataFrame(features_list)


def main():
    """Test feature building."""
    import pandas as pd

    # Load sample data
    historical = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/historical_matches.csv")
    standings = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv")
    fixtures = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv")

    builder = FeatureBuilder()

    # Build training features
    print("Building training features...")
    train_features = builder.build_training_features(historical, standings)
    train_features.to_csv(f"{config.DATA_PROCESSED_DIR}/train_features.csv", index=False)
    print(f"Training features shape: {train_features.shape}")
    print(f"Features: {list(train_features.columns)}")

    # Build prediction features
    print("\nBuilding prediction features...")
    pred_features = builder.build_prediction_features(fixtures, historical, standings)
    pred_features.to_csv(f"{config.DATA_PROCESSED_DIR}/prediction_features.csv", index=False)
    print(f"Prediction features shape: {pred_features.shape}")


if __name__ == '__main__':
    main()
