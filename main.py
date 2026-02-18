"""Main entry point for Premier League Prediction Model.

This script runs the complete pipeline:
1. Data collection
2. Feature engineering
3. Model training
4. Prediction generation
5. Season simulation
6. Visualization
"""

import os
import sys
import argparse
from datetime import datetime

import config


def setup_directories():
    """Create necessary directories."""
    dirs = [
        config.DATA_RAW_DIR,
        config.DATA_PROCESSED_DIR,
        config.PREDICTIONS_DIR,
        config.MODELS_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def run_data_collection(use_sample_data: bool = False):
    """Step 1: Collect or generate data."""
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION")
    print("="*80)

    from data.fetch_data import DataFetcher

    fetcher = DataFetcher()

    if use_sample_data:
        print("Using sample data...")
        standings, historical, fixtures = fetcher.load_or_generate_sample_data()
    else:
        print("Attempting to fetch real data from API...")
        standings = fetcher.fetch_current_standings()

        if standings.empty:
            print("API data unavailable, falling back to sample data...")
            standings, historical, fixtures = fetcher.load_or_generate_sample_data()
        else:
            historical = fetcher.fetch_historical_data()
            fixtures = fetcher.fetch_remaining_fixtures()

    # Save data
    standings.to_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv", index=False)
    historical.to_csv(f"{config.DATA_PROCESSED_DIR}/historical_matches.csv", index=False)
    fixtures.to_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv", index=False)

    print(f"\nData saved:")
    print(f"  - Standings: {len(standings)} teams")
    print(f"  - Historical matches: {len(historical)} matches")
    print(f"  - Remaining fixtures: {len(fixtures)} fixtures")

    return standings, historical, fixtures


def run_feature_engineering():
    """Step 2: Build features."""
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)

    from features.build_features import FeatureBuilder
    import pandas as pd

    # Load data
    historical = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/historical_matches.csv")
    standings = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv")
    fixtures = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv")

    builder = FeatureBuilder()

    # Build training features
    print("\nBuilding training features from historical matches...")
    train_features = builder.build_training_features(historical, standings)
    train_features.to_csv(f"{config.DATA_PROCESSED_DIR}/train_features.csv", index=False)
    print(f"Training features: {len(train_features)} samples, {len(train_features.columns)} features")

    # Build prediction features
    print("\nBuilding prediction features for remaining fixtures...")
    pred_features = builder.build_prediction_features(fixtures, historical, standings)
    pred_features.to_csv(f"{config.DATA_PROCESSED_DIR}/prediction_features.csv", index=False)
    print(f"Prediction features: {len(pred_features)} fixtures")

    return train_features, pred_features


def run_model_training():
    """Step 3: Train the prediction model."""
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80)

    from models.train_model import MatchPredictor
    import pandas as pd

    # Load features
    features_df = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/train_features.csv")
    print(f"Training on {len(features_df)} samples...")

    # Train model
    predictor = MatchPredictor()
    results = predictor.train(features_df)

    # Save model
    predictor.save_model()

    print("\nModel Training Summary:")
    print("-"*40)
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            print(f"{model_name:20} - Accuracy: {metrics['accuracy']:.4f}")

    return predictor


def run_predictions():
    """Step 4: Generate predictions for remaining fixtures."""
    print("\n" + "="*80)
    print("STEP 4: PREDICTION GENERATION")
    print("="*80)

    from models.predict import PredictionEngine

    engine = PredictionEngine()

    # Predict remaining fixtures
    predictions = engine.predict_remaining_fixtures()

    # Save predictions
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    predictions.to_csv(f"{config.PREDICTIONS_DIR}/match_predictions.csv", index=False)

    print(f"\nGenerated predictions for {len(predictions)} matches")
    print("\nSample Predictions:")
    print("-"*80)

    for _, pred in predictions.head(5).iterrows():
        print(f"\n{pred['home_team']} vs {pred['away_team']}")
        print(f"  Prediction: {pred['predicted_outcome']} "
              f"(H:{pred['prob_home_win']:.1%} D:{pred['prob_draw']:.1%} A:{pred['prob_away_win']:.1%})")
        print(f"  Expected Score: {pred['expected_home_goals']:.1f} - {pred['expected_away_goals']:.1f}")

    return predictions


def run_simulation():
    """Step 5: Monte Carlo season simulation."""
    print("\n" + "="*80)
    print("STEP 5: SEASON SIMULATION")
    print("="*80)

    from simulation.season_simulator import SeasonSimulator
    import pandas as pd

    # Load data
    standings = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv")
    fixtures = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv")
    predictions = pd.read_csv(f"{config.PREDICTIONS_DIR}/match_predictions.csv")

    print(f"Running {config.N_SIMULATIONS} season simulations...")
    print(f"Current standings: {len(standings)} teams")
    print(f"Remaining fixtures: {len(fixtures)} matches")

    # Run simulation
    simulator = SeasonSimulator(standings, fixtures, predictions,
                                   n_simulations=config.N_SIMULATIONS)
    final_standings, results = simulator.get_final_standings_prediction()

    # Save results
    final_standings.to_csv(f"{config.PREDICTIONS_DIR}/final_standings.csv", index=False)

    print("\n" + "="*80)
    print("SIMULATION RESULTS - Predicted Final Standings")
    print("="*80)
    print(f"{'Pos':<4} {'Team':<25} {'Pts':<6} {'ExpPts':<8} "
          f"{'Champ%':<8} {'Top4%':<8} {'Top6%':<8} {'Rel%':<8}")
    print("-"*80)

    for _, team in final_standings.iterrows():
        print(f"{team['predicted_position']:<4} {team['team_name']:<25} "
              f"{team['current_points']:<6} {team['expected_final_points']:<8} "
              f"{team['champion_probability']:<8} {team['top4_probability']:<8} "
              f"{team['top6_probability']:<8} {team['relegation_probability']:<8}")

    # Print key insights
    print("\n" + "="*80)
    print("KEY PREDICTIONS")
    print("="*80)

    # Champion
    champ = final_standings.nlargest(1, 'champion_probability').iloc[0]
    print(f"\n🏆 CHAMPION: {champ['team_name']} ({champ['champion_probability']:.1f}%)")

    # Top 4 race
    print("\n📊 TOP 4 RACE:")
    top4 = final_standings.nlargest(6, 'top4_probability')
    for _, team in top4.iterrows():
        if team['top4_probability'] > 1:
            status = "✓" if team['top4_probability'] > 50 else "?"
            print(f"   {status} {team['team_name']}: {team['top4_probability']:.1f}%")

    # Relegation battle
    print("\n⚠️  RELEGATION BATTLE:")
    rel = final_standings.nlargest(5, 'relegation_probability')
    for _, team in rel.iterrows():
        if team['relegation_probability'] > 1:
            status = "✗" if team['relegation_probability'] > 50 else "?"
            print(f"   {status} {team['team_name']}: {team['relegation_probability']:.1f}%")

    return final_standings, results


def run_visualization():
    """Step 6: Generate visualizations."""
    print("\n" + "="*80)
    print("STEP 6: VISUALIZATION")
    print("="*80)

    from visualization.plot_standings import create_all_visualizations

    create_all_visualizations()

    print(f"\nVisualizations saved to {config.PREDICTIONS_DIR}/")


def run_full_pipeline(args):
    """Run the complete pipeline."""
    start_time = datetime.now()

    print("="*80)
    print("PREMIER LEAGUE 2024/25 PREDICTION MODEL")
    print("="*80)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    setup_directories()

    # Step 1: Data Collection
    standings, historical, fixtures = run_data_collection(args.use_sample_data)

    # Step 2: Feature Engineering
    train_features, pred_features = run_feature_engineering()

    # Step 3: Model Training
    predictor = run_model_training()

    # Step 4: Predictions
    predictions = run_predictions()

    # Step 5: Simulation
    final_standings, results = run_simulation()

    # Step 6: Visualization
    if not args.skip_viz:
        run_visualization()

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print("\nOutput files:")
    print(f"  - {config.PREDICTIONS_DIR}/final_standings.csv")
    print(f"  - {config.PREDICTIONS_DIR}/match_predictions.csv")
    print(f"  - {config.PREDICTIONS_DIR}/*.png (visualizations)")


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description='Premier League Prediction Model 2024/25'
    )
    parser.add_argument(
        '--use-sample-data', action='store_true',
        help='Use sample data instead of fetching from API'
    )
    parser.add_argument(
        '--skip-viz', action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--step', type=str, choices=['data', 'features', 'train', 'predict', 'simulate', 'viz', 'all'],
        default='all', help='Run specific step only'
    )

    args = parser.parse_args()

    setup_directories()

    if args.step == 'all':
        run_full_pipeline(args)
    elif args.step == 'data':
        run_data_collection(args.use_sample_data)
    elif args.step == 'features':
        run_feature_engineering()
    elif args.step == 'train':
        run_model_training()
    elif args.step == 'predict':
        run_predictions()
    elif args.step == 'simulate':
        run_simulation()
    elif args.step == 'viz':
        run_visualization()


if __name__ == '__main__':
    main()
