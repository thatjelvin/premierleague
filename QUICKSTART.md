# Premier League Prediction Model

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Get API key from https://www.football-data.org/
   - Create `.env` file with: `FOOTBALL_DATA_API_KEY=your_key`

3. Run the model:
```bash
python main.py
```

Or with sample data (no API key needed):
```bash
python main.py --use-sample-data
```

## Quick Start

Run the full pipeline:
```bash
python main.py
```

## Output

Results are saved to `predictions/`:
- `final_standings.csv` - Predicted final table
- `match_predictions.csv` - Match outcome probabilities
- `*.png` - Visualization charts
