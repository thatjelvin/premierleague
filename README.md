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

## 2024/25 Season Predictions

Based on 10,000 Monte Carlo simulations using ensemble ML models (XGBoost, Random Forest, Gradient Boosting).

### 🏆 Predicted Final Standings

| Pos | Team                    | Current Pts | Expected Pts | Champion % | Top 4 % | Top 6 % | Relegation % |
|-----|-------------------------|-------------|--------------|------------|---------|---------|--------------|
| 1   | Liverpool               | 59          | 81.9         | 99.4%      | 100.0%  | 100.0%  | 0.0%         |
| 2   | Manchester City         | 47          | 65.3         | 0.4%       | 91.5%   | 98.0%   | 0.0%         |
| 3   | Manchester United       | 45          | 63.7         | 0.1%       | 86.5%   | 97.2%   | 0.0%         |
| 4   | Newcastle United        | 40          | 58.4         | 0.0%       | 36.9%   | 69.0%   | 0.0%         |
| 5   | Nottingham Forest       | 39          | 56.9         | 0.0%       | 24.1%   | 56.2%   | 0.0%         |
| 6   | Arsenal                 | 41          | 55.8         | 0.0%       | 19.4%   | 48.9%   | 0.0%         |
| 7   | Crystal Palace          | 32          | 55.1         | 0.0%       | 15.6%   | 41.2%   | 0.0%         |
| 8   | Chelsea                 | 37          | 53.6         | 0.0%       | 7.8%    | 26.0%   | 0.0%         |
| 9   | Tottenham Hotspur       | 41          | 53.5         | 0.0%       | 9.7%    | 28.6%   | 0.0%         |
| 10  | Brighton                | 35          | 52.0         | 0.0%       | 4.8%    | 16.3%   | 0.0%         |
| 11  | Brentford               | 30          | 50.9         | 0.0%       | 3.4%    | 14.3%   | 0.1%         |
| 12  | Ipswich Town            | 27          | 47.4         | 0.0%       | 0.4%    | 2.7%    | 0.9%         |
| 13  | Wolverhampton           | 29          | 45.4         | 0.0%       | 0.1%    | 1.1%    | 1.2%         |
| 14  | West Bromwich Albion    | 26          | 44.4         | 0.0%       | 0.0%    | 0.5%    | 1.5%         |
| 15  | AFC Bournemouth         | 22          | 40.4         | 0.0%       | 0.0%    | 0.1%    | 12.1%        |
| 16  | Sheffield United        | 25          | 39.8         | 0.0%       | 0.0%    | 0.1%    | 15.2%        |
| 17  | Southampton             | 23          | 39.5         | 0.0%       | 0.0%    | 0.0%    | 16.4%        |
| 18  | Norwich City            | 18          | 35.8         | 0.0%       | 0.0%    | 0.0%    | 52.6%        |

### 📊 Key Predictions

#### Title Race
- **Liverpool** are overwhelming favorites with a **99.4%** chance of winning the Premier League
- **Manchester City** have a slim **0.4%** chance but are virtually guaranteed Top 4
- The title race appears all but decided with Liverpool's commanding lead

#### Champions League (Top 4)
The Top 4 race is competitive with several teams in contention:
- **Liverpool**: 100.0% (Secured)
- **Manchester City**: 91.5% (Strong favorite)
- **Manchester United**: 86.5% (Strong favorite)
- **Newcastle United**: 36.9% (Fighting for 4th)
- **Nottingham Forest**: 24.1% (Outsider)
- **Arsenal**: 19.4% (Long shot)

#### Relegation Battle
The relegation battle features 5 teams at serious risk:
- **Norwich City**: 52.6% (Most likely to go down)
- **Southampton**: 16.4%
- **Sheffield United**: 15.2%
- **AFC Bournemouth**: 12.1%
- **West Bromwich Albion**: 1.5%

### 📈 Model Performance

The ensemble model achieved the following validation performance:
- **Random Forest**: 46.6% accuracy (best performer)
- **XGBoost**: 42.0% accuracy
- **Gradient Boosting**: 41.1% accuracy
- **Ensemble**: Weighted combination (50% XGBoost, 30% RF, 20% GB)

**Top Features:**
1. Away team rest days
2. Head-to-head home win percentage
3. Home team PPG at home
4. Away team form (recent wins)
5. Home team goals per game

### 🎯 Sample Match Predictions

| Home Team | Away Team | Predicted Result | Home Win | Draw | Away Win |
|-----------|-----------|------------------|----------|------|----------|
| Liverpool | Man City | H | 65.9% | 15.4% | 18.6% |
| Brighton | Man Utd | H | 69.7% | 17.8% | 12.5% |
| Crystal Palace | Wolves | H | 64.2% | 17.5% | 18.3% |
| Ipswich Town | Sheffield Utd | H | 72.8% | 9.4% | 17.7% |
| Newcastle | West Brom | H | 66.9% | 17.7% | 15.5% |

### 📊 Visualization Outputs

The model generates the following visualizations (saved in `predictions/` directory):

1. **position_probabilities.png** - Heatmap showing probability of each team finishing in each position
2. **final_standings_chart.png** - Bar chart of predicted final standings with confidence intervals
3. **outcome_probabilities.png** - Stacked bar chart of key outcome probabilities
4. **fixture_difficulty.png** - Heatmap of remaining fixture difficulty by team
5. **points_distribution.png** - Distribution of simulated final points for top teams

## License

MIT
