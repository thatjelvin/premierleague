"""Model training module for Premier League prediction."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import os
import sys
from typing import Dict, Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MatchPredictor:
    """Machine learning model for predicting match outcomes."""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Prepare features for model input."""
        # Select numeric features (exclude IDs and target)
        exclude_cols = ['outcome', 'match_id', 'home_team_id', 'away_team_id',
                       'home_team', 'away_team', 'season', 'matchday']

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols

        X = df[feature_cols].values

        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0)

        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def train(self, features_df: pd.DataFrame, validation_size: float = 0.2) -> Dict:
        """Train multiple models and evaluate performance."""
        print("Preparing features...")

        # Split by season for time-series aware validation
        seasons = sorted(features_df['season'].unique())
        val_seasons = seasons[-2:] if len(seasons) >= 2 else [seasons[-1]]

        train_df = features_df[~features_df['season'].isin(val_seasons)]
        val_df = features_df[features_df['season'].isin(val_seasons)]

        if len(val_df) == 0:
            # Fallback to random split
            train_df, val_df = train_test_split(
                features_df, test_size=validation_size,
                random_state=config.RANDOM_STATE, stratify=features_df['outcome']
            )

        X_train = self._prepare_features(train_df, fit_scaler=True)
        y_train = train_df['outcome'].values

        X_val = self._prepare_features(val_df, fit_scaler=False)
        y_val = val_df['outcome'].values

        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Features: {len(self.feature_columns)}")

        # Train multiple models
        results = {}

        # 1. XGBoost Classifier
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            random_state=config.RANDOM_STATE,
            eval_metric='mlogloss'
        )

        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        xgb_pred = xgb_model.predict(X_val)
        xgb_proba = xgb_model.predict_proba(X_val)
        xgb_acc = accuracy_score(y_val, xgb_pred)
        xgb_loss = log_loss(y_val, xgb_proba)

        print(f"XGBoost - Accuracy: {xgb_acc:.4f}, Log Loss: {xgb_loss:.4f}")

        self.models['xgboost'] = xgb_model
        results['xgboost'] = {
            'accuracy': xgb_acc,
            'log_loss': xgb_loss,
            'feature_importance': dict(zip(
                self.feature_columns,
                xgb_model.feature_importances_
            ))
        }

        # 2. Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=config.RANDOM_STATE
        )

        # Calibrate for probability estimates
        rf_calibrated = CalibratedClassifierCV(
            rf_model, method='sigmoid', cv=5
        )
        rf_calibrated.fit(X_train, y_train)

        rf_pred = rf_calibrated.predict(X_val)
        rf_proba = rf_calibrated.predict_proba(X_val)
        rf_acc = accuracy_score(y_val, rf_pred)
        rf_loss = log_loss(y_val, rf_proba)

        print(f"Random Forest - Accuracy: {rf_acc:.4f}, Log Loss: {rf_loss:.4f}")

        self.models['random_forest'] = rf_calibrated
        results['random_forest'] = {
            'accuracy': rf_acc,
            'log_loss': rf_loss
        }

        # 3. Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE
        )
        gb_model.fit(X_train, y_train)

        gb_pred = gb_model.predict(X_val)
        gb_proba = gb_model.predict_proba(X_val)
        gb_acc = accuracy_score(y_val, gb_pred)
        gb_loss = log_loss(y_val, gb_proba)

        print(f"Gradient Boosting - Accuracy: {gb_acc:.4f}, Log Loss: {gb_loss:.4f}")

        self.models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'accuracy': gb_acc,
            'log_loss': gb_loss
        }

        # Print classification report for best model (XGBoost usually)
        print("\n" + "="*50)
        print("XGBoost Classification Report:")
        print("="*50)
        print(classification_report(y_val, xgb_pred,
                                   target_names=['Home Win', 'Draw', 'Away Win']))

        # Print top features
        print("\n" + "="*50)
        print("Top 10 Important Features:")
        print("="*50)
        importance = results['xgboost']['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:10]:
            print(f"  {feature}: {imp:.4f}")

        self.is_fitted = True
        return results

    def predict_match(self, features: Dict) -> Dict:
        """Predict outcome probabilities for a single match."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        # Create feature vector
        feature_vector = np.array([[features.get(col, 0) for col in self.feature_columns]])
        feature_vector = self.scaler.transform(feature_vector)

        # Get predictions from all models
        predictions = {}

        # XGBoost prediction
        xgb_proba = self.models['xgboost'].predict_proba(feature_vector)[0]
        predictions['xgboost'] = {
            'home_win': xgb_proba[0],
            'draw': xgb_proba[1],
            'away_win': xgb_proba[2]
        }

        # Random Forest prediction
        rf_proba = self.models['random_forest'].predict_proba(feature_vector)[0]
        predictions['random_forest'] = {
            'home_win': rf_proba[0],
            'draw': rf_proba[1],
            'away_win': rf_proba[2]
        }

        # Gradient Boosting prediction
        gb_proba = self.models['gradient_boosting'].predict_proba(feature_vector)[0]
        predictions['gradient_boosting'] = {
            'home_win': gb_proba[0],
            'draw': gb_proba[1],
            'away_win': gb_proba[2]
        }

        # Ensemble prediction (weighted average, favoring XGBoost)
        ensemble_proba = (
            0.5 * xgb_proba +
            0.3 * rf_proba +
            0.2 * gb_proba
        )

        predictions['ensemble'] = {
            'home_win': ensemble_proba[0],
            'draw': ensemble_proba[1],
            'away_win': ensemble_proba[2]
        }

        # Most likely outcome
        outcomes = ['Home Win', 'Draw', 'Away Win']
        predicted_outcome = outcomes[np.argmax(ensemble_proba)]
        confidence = np.max(ensemble_proba)

        predictions['predicted_outcome'] = predicted_outcome
        predictions['confidence'] = confidence
        predictions['expected_home_goals'] = ensemble_proba[0] * 2.5 + ensemble_proba[1] * 1.2
        predictions['expected_away_goals'] = ensemble_proba[2] * 2.0 + ensemble_proba[1] * 1.2

        return predictions

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict outcomes for multiple matches."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        X = self._prepare_features(features_df, fit_scaler=False)

        results = []

        for idx, x in enumerate(X):
            x = x.reshape(1, -1)
            row = features_df.iloc[idx]

            # Get ensemble probabilities
            xgb_proba = self.models['xgboost'].predict_proba(x)[0]
            rf_proba = self.models['random_forest'].predict_proba(x)[0]
            gb_proba = self.models['gradient_boosting'].predict_proba(x)[0]

            ensemble_proba = 0.5 * xgb_proba + 0.3 * rf_proba + 0.2 * gb_proba

            outcomes = ['H', 'D', 'A']
            predicted = outcomes[np.argmax(ensemble_proba)]

            results.append({
                'match_id': row.get('match_id', idx),
                'home_team': row.get('home_team', 'Unknown'),
                'away_team': row.get('away_team', 'Unknown'),
                'home_team_id': row.get('home_team_id', 0),
                'away_team_id': row.get('away_team_id', 0),
                'matchday': row.get('matchday', 0),
                'prob_home_win': ensemble_proba[0],
                'prob_draw': ensemble_proba[1],
                'prob_away_win': ensemble_proba[2],
                'predicted_outcome': predicted,
                'confidence': np.max(ensemble_proba),
                'expected_home_goals': ensemble_proba[0] * 2.5 + ensemble_proba[1] * 1.2,
                'expected_away_goals': ensemble_proba[2] * 2.0 + ensemble_proba[1] * 1.2
            })

        return pd.DataFrame(results)

    def save_model(self, path: str = None):
        """Save trained model to disk."""
        if path is None:
            path = config.MODELS_DIR

        os.makedirs(path, exist_ok=True)

        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, f"{path}/match_predictor.pkl")
        print(f"Model saved to {path}/match_predictor.pkl")

    def load_model(self, path: str = None):
        """Load trained model from disk."""
        if path is None:
            path = config.MODELS_DIR

        model_path = f"{path}/match_predictor.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {model_path}")


def main():
    """Train model on sample data."""
    # Load features
    features_path = f"{config.DATA_PROCESSED_DIR}/train_features.csv"

    if not os.path.exists(features_path):
        print("Features not found. Please run feature engineering first.")
        print("Run: python data/fetch_data.py")
        print("     python features/build_features.py")
        return

    features_df = pd.read_csv(features_path)
    print(f"Loaded {len(features_df)} training samples")

    # Train model
    predictor = MatchPredictor()
    results = predictor.train(features_df)

    # Save model
    predictor.save_model()

    print("\nModel training complete!")


if __name__ == '__main__':
    main()
