# Premier League Prediction Model

A machine learning model to predict the 2024/25 Premier League season outcomes using Monte Carlo simulation.

## Features

- **Data Collection**: Fetch current standings, historical matches, and remaining fixtures from football-data.org API
- **Feature Engineering**: Comprehensive match features including form, head-to-head, home/away performance, and league position
- **Machine Learning**: Ensemble of XGBoost, Random Forest, and Gradient Boosting models
- **Monte Carlo Simulation**: 10,000+ season simulations for robust probability estimates
- **Visualizations**: Position probability heatmaps, outcome charts, and fixture difficulty analysis

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Set up API key (get free key from football-data.org)
export FOOTBALL_DATA_API_KEY="your_api_key"
```

## Usage

### Run Full Pipeline

```bash
python main.py
```

### Run Specific Steps

```bash
# Data collection only
python main.py --step data

# Feature engineering
python main.py --step features

# Model training
python main.py --step train

# Generate predictions
python main.py --step predict

# Run simulation
python main.py --step simulate

# Generate visualizations
python main.py --step viz
```

### Using Sample Data (no API key required)

```bash
python main.py --use-sample-data
```

## Project Structure

```
premier-league-predictor/
├── data/
│   ├── fetch_data.py      # Data collection module
│   ├── raw/               # Cached API responses
│   └── processed/         # Cleaned datasets
├── features/
│   └── build_features.py  # Feature engineering
├── models/
│   ├── train_model.py     # Model training
│   └── predict.py         # Prediction utilities
├── simulation/
│   └── season_simulator.py # Monte Carlo simulation
├── visualization/
│   └── plot_standings.py  # Charts and visualizations
├── predictions/           # Output directory
│   ├── final_standings.csv
│   ├── match_predictions.csv
│   └── *.png              # Visualization charts
├── config.py              # Configuration settings
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

## Model Features

- **Team Form**: Last 5 and 10 matches points
- **Home/Away Form**: Venue-specific performance
- **Season Statistics**: Points per game, goals scored/conceded
- **Head-to-Head**: Historical results between teams
- **League Position**: Current standing and position difference
- **Rest Days**: Days since last match (fatigue factor)
- **Match Context**: Matchday, season progression

## Outputs

1. **Final Standings Prediction**: CSV with predicted positions and probabilities
2. **Match Predictions**: Individual match outcome probabilities
3. **Position Probability Heatmap**: Likelihood of each team finishing in each position
4. **Outcome Probabilities**: Stacked bar chart of key outcomes
5. **Fixture Difficulty**: Remaining schedule difficulty by team
6. **Points Distribution**: Distribution of simulated final points

## Key Metrics

- **Champion Probability**: Likelihood of winning the title
- **Top 4 Probability**: Champions League qualification chance
- **Top 6 Probability**: European competition qualification
- **Relegation Probability**: Risk of relegation

## License

MIT
