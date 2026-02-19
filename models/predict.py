"""Prediction utilities for match outcomes."""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.train_model import MatchPredictor


class PredictionEngine:
    """High-level prediction interface."""

    def __init__(self, model_path: str = None):
        self.predictor = MatchPredictor()

        if model_path is None:
            model_path = config.MODELS_DIR

        try:
            self.predictor.load_model(model_path)
        except FileNotFoundError:
            print("Model not found. Please train model first.")
            raise

    def predict_remaining_fixtures(self, fixtures_path: str = None,
                                    features_path: str = None) -> pd.DataFrame:
        """Predict outcomes for all remaining fixtures."""
        if fixtures_path is None:
            fixtures_path = f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv"
        if features_path is None:
            features_path = f"{config.DATA_PROCESSED_DIR}/prediction_features.csv"

        # Load prediction features
        pred_features = pd.read_csv(features_path)
        fixtures = pd.read_csv(fixtures_path)

        print(f"Predicting {len(pred_features)} matches...")

        # Get predictions
        predictions = self.predictor.predict_batch(pred_features)

        # Merge with fixture details
        results = predictions.merge(
            fixtures[['match_id', 'date', 'matchday']],
            on='match_id',
            how='left'
        )

        return results

    def get_fixture_difficulty(self, team_id: int, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fixture difficulty for a team."""
        team_fixtures = predictions_df[
            (predictions_df['home_team_id'] == team_id) |
            (predictions_df['away_team_id'] == team_id)
        ].copy()

        def calc_difficulty(row):
            if row['home_team_id'] == team_id:
                # Playing at home
                return 1 - row['prob_home_win'] - 0.5 * row['prob_draw']
            else:
                # Playing away
                return 1 - row['prob_away_win'] - 0.5 * row['prob_draw']

        team_fixtures['difficulty'] = team_fixtures.apply(calc_difficulty, axis=1)
        team_fixtures['difficulty_rating'] = pd.cut(
            team_fixtures['difficulty'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Easy', 'Medium', 'Hard', 'Very Hard']
        )

        return team_fixtures.sort_values('matchday')

    def predict_match(self, home_team_id: int, away_team_id: int,
                      current_standings: pd.DataFrame,
                      historical_df: pd.DataFrame) -> Dict:
        """Predict a single hypothetical match."""
        from features.build_features import FeatureBuilder

        # Build features for this hypothetical match
        builder = FeatureBuilder()

        # Create a mock fixture
        fixture = pd.Series({
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'date': pd.Timestamp.now().isoformat(),
            'matchday': 30,
            'match_id': -1
        })

        features = builder.build_features_for_match(
            historical_df[historical_df['season'] == config.CURRENT_SEASON],
            fixture,
            current_standings
        )

        # Predict
        prediction = self.predictor.predict_match(features)

        # Get team names
        home_name = current_standings[
            current_standings['team_id'] == home_team_id
        ]['team_name'].values
        away_name = current_standings[
            current_standings['team_id'] == away_team_id
        ]['team_name'].values

        home_name = home_name[0] if len(home_name) > 0 else 'Unknown'
        away_name = away_name[0] if len(away_name) > 0 else 'Unknown'

        return {
            'home_team': home_name,
            'away_team': away_name,
            'probabilities': prediction['ensemble'],
            'predicted_outcome': prediction['predicted_outcome'],
            'confidence': prediction['confidence'],
            'expected_score': (
                round(prediction['expected_home_goals'], 1),
                round(prediction['expected_away_goals'], 1)
            )
        }


def main():
    """Test predictions."""
    engine = PredictionEngine()

    # Predict remaining fixtures
    predictions = engine.predict_remaining_fixtures()

    # Save predictions
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    predictions.to_csv(f"{config.PREDICTIONS_DIR}/match_predictions.csv", index=False)

    print("\nSample Predictions:")
    print("="*80)
    for _, pred in predictions.head(10).iterrows():
        print(f"\n{pred['home_team']} vs {pred['away_team']} (Matchday {pred['matchday']})")
        print(f"  Predicted: {pred['predicted_outcome']}")
        print(f"  Probabilities: H={pred['prob_home_win']:.2%}, "
              f"D={pred['prob_draw']:.2%}, A={pred['prob_away_win']:.2%}")
        print(f"  Expected Score: {pred['expected_home_goals']:.1f} - {pred['expected_away_goals']:.1f}")
        print(f"  Confidence: {pred['confidence']:.2%}")

    print(f"\nSaved {len(predictions)} predictions to {config.PREDICTIONS_DIR}/match_predictions.csv")


if __name__ == '__main__':
    main()
