"""Configuration settings for Premier League Predictor."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
FOOTBALL_DATA_API_KEY = os.getenv('FOOTBALL_DATA_API_KEY', '')
FOOTBALL_DATA_BASE_URL = 'https://api.football-data.org/v4'

# Rate limiting (10 calls per minute for free tier)
API_CALL_DELAY = 7  # seconds between calls

# Data paths
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
PREDICTIONS_DIR = 'predictions'
MODELS_DIR = 'models'

# Season configuration
CURRENT_SEASON = 2024
TRAINING_SEASONS = [2020, 2021, 2022, 2023, 2024]
PREMIER_LEAGUE_CODE = 'PL'

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SIMULATIONS = 10000

# Feature configuration
FORM_WINDOW_SHORT = 5
FORM_WINDOW_LONG = 10
HOME_ADVANTAGE_FACTOR = 1.36  # Historical home win rate

# Points system
WIN_POINTS = 3
DRAW_POINTS = 1
LOSS_POINTS = 0
