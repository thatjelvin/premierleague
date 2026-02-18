"""Module to fetch Premier League data from football-data.org API."""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from typing import Optional, Dict, List
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FootballDataAPI:
    """Client for the football-data.org API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.FOOTBALL_DATA_API_KEY
        self.base_url = config.FOOTBALL_DATA_BASE_URL
        self.headers = {'X-Auth-Token': self.api_key} if self.api_key else {}
        self.last_call_time = 0

    def _rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_call_time
        if elapsed < config.API_CALL_DELAY:
            time.sleep(config.API_CALL_DELAY - elapsed)
        self.last_call_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a rate-limited API request."""
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e.response, 'status_code') and e.response.status_code == 403:
                print("Note: football-data.org requires an API key for most endpoints.")
                print("Get a free key at: https://www.football-data.org/")
            return {}

    def get_competitions(self) -> Dict:
        """Get available competitions."""
        return self._make_request('competitions')

    def get_standings(self, competition_code: str = 'PL', season: Optional[int] = None) -> Dict:
        """Get current standings for a competition."""
        endpoint = f"competitions/{competition_code}/standings"
        params = {'season': season} if season else None
        return self._make_request(endpoint, params)

    def get_matches(self, competition_code: str = 'PL', season: Optional[int] = None,
                    date_from: Optional[str] = None, date_to: Optional[str] = None,
                    status: Optional[str] = None) -> Dict:
        """Get matches for a competition."""
        endpoint = f"competitions/{competition_code}/matches"
        params = {}
        if season:
            params['season'] = season
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        if status:
            params['status'] = status
        return self._make_request(endpoint, params)

    def get_team_matches(self, team_id: int, season: Optional[int] = None) -> Dict:
        """Get matches for a specific team."""
        endpoint = f"teams/{team_id}/matches"
        params = {'season': season} if season else None
        return self._make_request(endpoint, params)

    def get_team(self, team_id: int) -> Dict:
        """Get team information."""
        return self._make_request(f"teams/{team_id}")


class DataFetcher:
    """High-level data fetching and caching manager."""

    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        self.api = FootballDataAPI(api_key)
        self.use_cache = use_cache
        self.raw_dir = config.DATA_RAW_DIR
        self.processed_dir = config.DATA_PROCESSED_DIR

        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def _cache_path(self, name: str) -> str:
        """Get cache file path."""
        return os.path.join(self.raw_dir, f"{name}.json")

    def _load_cache(self, name: str) -> Optional[Dict]:
        """Load data from cache if it exists."""
        if not self.use_cache:
            return None
        cache_path = self._cache_path(name)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def _save_cache(self, name: str, data: Dict):
        """Save data to cache."""
        cache_path = self._cache_path(name)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)

    def fetch_current_standings(self, use_cache: bool = True) -> pd.DataFrame:
        """Fetch current Premier League standings."""
        cache_name = f"standings_{config.CURRENT_SEASON}"

        if use_cache:
            data = self._load_cache(cache_name)
            if data is None:
                data = self.api.get_standings(config.PREMIER_LEAGUE_CODE, config.CURRENT_SEASON)
                self._save_cache(cache_name, data)
        else:
            data = self.api.get_standings(config.PREMIER_LEAGUE_CODE, config.CURRENT_SEASON)
            self._save_cache(cache_name, data)

        if not data or 'standings' not in data:
            print("Warning: Could not fetch standings data")
            return pd.DataFrame()

        # Parse standings
        table = data['standings'][0]['table']
        teams = []
        for team_data in table:
            teams.append({
                'position': team_data['position'],
                'team_id': team_data['team']['id'],
                'team_name': team_data['team']['name'],
                'short_name': team_data['team']['shortName'],
                'played': team_data['playedGames'],
                'won': team_data['won'],
                'drawn': team_data['draw'],
                'lost': team_data['lost'],
                'goals_for': team_data['goalsFor'],
                'goals_against': team_data['goalsAgainst'],
                'goal_difference': team_data['goalDifference'],
                'points': team_data['points'],
                'form': team_data.get('form', ''),
            })

        return pd.DataFrame(teams)

    def fetch_season_matches(self, season: int, use_cache: bool = True) -> pd.DataFrame:
        """Fetch all matches for a specific season."""
        cache_name = f"matches_{season}"

        if use_cache:
            data = self._load_cache(cache_name)
            if data is None:
                data = self.api.get_matches(config.PREMIER_LEAGUE_CODE, season=season)
                self._save_cache(cache_name, data)
        else:
            data = self.api.get_matches(config.PREMIER_LEAGUE_CODE, season=season)
            self._save_cache(cache_name, data)

        if not data or 'matches' not in data:
            print(f"Warning: Could not fetch matches for season {season}")
            return pd.DataFrame()

        matches = []
        for match in data['matches']:
            if match['status'] == 'FINISHED':
                matches.append({
                    'match_id': match['id'],
                    'season': season,
                    'date': match['utcDate'],
                    'matchday': match['matchday'],
                    'home_team_id': match['homeTeam']['id'],
                    'home_team': match['homeTeam']['name'],
                    'away_team_id': match['awayTeam']['id'],
                    'away_team': match['awayTeam']['name'],
                    'home_goals': match['score']['fullTime'].get('home', 0) or 0,
                    'away_goals': match['score']['fullTime'].get('away', 0) or 0,
                    'winner': match['score'].get('winner', 'DRAW'),
                    'status': match['status']
                })

        return pd.DataFrame(matches)

    def fetch_remaining_fixtures(self, use_cache: bool = True) -> pd.DataFrame:
        """Fetch remaining fixtures for current season."""
        cache_name = f"fixtures_remaining_{config.CURRENT_SEASON}"

        # Try to get scheduled matches
        today = datetime.now().strftime('%Y-%m-%d')

        if use_cache:
            data = self._load_cache(cache_name)
            if data is None:
                data = self.api.get_matches(
                    config.PREMIER_LEAGUE_CODE,
                    season=config.CURRENT_SEASON,
                    status='SCHEDULED'
                )
                self._save_cache(cache_name, data)
        else:
            data = self.api.get_matches(
                config.PREMIER_LEAGUE_CODE,
                season=config.CURRENT_SEASON,
                status='SCHEDULED'
            )
            self._save_cache(cache_name, data)

        if not data or 'matches' not in data:
            print("Warning: Could not fetch remaining fixtures")
            return pd.DataFrame()

        fixtures = []
        for match in data['matches']:
            fixtures.append({
                'match_id': match['id'],
                'season': config.CURRENT_SEASON,
                'date': match['utcDate'],
                'matchday': match['matchday'],
                'home_team_id': match['homeTeam']['id'],
                'home_team': match['homeTeam']['name'],
                'away_team_id': match['awayTeam']['id'],
                'away_team': match['awayTeam']['name'],
                'status': match['status']
            })

        return pd.DataFrame(fixtures)

    def fetch_historical_data(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch historical match data for multiple seasons."""
        if seasons is None:
            seasons = config.TRAINING_SEASONS

        all_matches = []
        for season in seasons:
            print(f"Fetching data for season {season}/{season+1}...")
            matches = self.fetch_season_matches(season)
            if not matches.empty:
                all_matches.append(matches)

        if all_matches:
            return pd.concat(all_matches, ignore_index=True)
        return pd.DataFrame()

    def load_or_generate_sample_data(self) -> tuple:
        """Generate sample data if API is not available."""
        print("Generating sample data for demonstration...")

        # Sample team data
        teams_data = {
            'team_id': [64, 65, 66, 67, 68, 73, 76, 349, 351, 354, 356, 397, 402, 563, 1044, 328, 340, 394],
            'team_name': ['Liverpool', 'Manchester City', 'Manchester United', 'Tottenham',
                         'Norwich', 'Tottenham', 'West Brom', 'Ipswich Town', 'Nottingham Forest',
                         'Crystal Palace', 'Sheffield United', 'Brighton', 'Brentford', 'AFC Bournemouth',
                         'Newcastle United', 'Chelsea', 'Southampton', 'Wolverhampton'],
            'short_name': ['LIV', 'MCI', 'MUN', 'TOT', 'NOR', 'TOT', 'WBA', 'IPS', 'NOT',
                          'CRY', 'SHE', 'BHA', 'BRE', 'BOU', 'NEW', 'CHE', 'SOU', 'WOL']
        }

        # Create sample standings (simulated current table)
        standings = pd.DataFrame([
            {'position': 1, 'team_id': 64, 'team_name': 'Liverpool', 'short_name': 'LIV',
             'played': 25, 'won': 18, 'drawn': 5, 'lost': 2, 'goals_for': 60, 'goals_against': 24,
             'goal_difference': 36, 'points': 59, 'form': 'WWWDW'},
            {'position': 2, 'team_id': 65, 'team_name': 'Manchester City', 'short_name': 'MCI',
             'played': 25, 'won': 14, 'drawn': 5, 'lost': 6, 'goals_for': 55, 'goals_against': 30,
             'goal_difference': 25, 'points': 47, 'form': 'WDLWW'},
            {'position': 3, 'team_id': 66, 'team_name': 'Manchester United', 'short_name': 'MUN',
             'played': 25, 'won': 13, 'drawn': 6, 'lost': 6, 'goals_for': 45, 'goals_against': 32,
             'goal_difference': 13, 'points': 45, 'form': 'WDWWW'},
            {'position': 4, 'team_id': 67, 'team_name': 'Tottenham Hotspur', 'short_name': 'TOT',
             'played': 25, 'won': 12, 'drawn': 5, 'lost': 8, 'goals_for': 50, 'goals_against': 35,
             'goal_difference': 15, 'points': 41, 'form': 'LWWDL'},
            {'position': 5, 'team_id': 73, 'team_name': 'Arsenal', 'short_name': 'ARS',
             'played': 25, 'won': 11, 'drawn': 8, 'lost': 6, 'goals_for': 48, 'goals_against': 34,
             'goal_difference': 14, 'points': 41, 'form': 'DWDWD'},
            {'position': 6, 'team_id': 1044, 'team_name': 'Newcastle United', 'short_name': 'NEW',
             'played': 25, 'won': 12, 'drawn': 4, 'lost': 9, 'goals_for': 42, 'goals_against': 38,
             'goal_difference': 4, 'points': 40, 'form': 'WWLLW'},
            {'position': 7, 'team_id': 351, 'team_name': 'Nottingham Forest', 'short_name': 'NFO',
             'played': 25, 'won': 11, 'drawn': 6, 'lost': 8, 'goals_for': 38, 'goals_against': 35,
             'goal_difference': 3, 'points': 39, 'form': 'DWWLW'},
            {'position': 8, 'team_id': 328, 'team_name': 'Chelsea', 'short_name': 'CHE',
             'played': 25, 'won': 10, 'drawn': 7, 'lost': 8, 'goals_for': 40, 'goals_against': 38,
             'goal_difference': 2, 'points': 37, 'form': 'WDLWD'},
            {'position': 9, 'team_id': 397, 'team_name': 'Brighton', 'short_name': 'BHA',
             'played': 25, 'won': 9, 'drawn': 8, 'lost': 8, 'goals_for': 38, 'goals_against': 37,
             'goal_difference': 1, 'points': 35, 'form': 'DDWDL'},
            {'position': 10, 'team_id': 354, 'team_name': 'Crystal Palace', 'short_name': 'CRY',
             'played': 25, 'won': 8, 'drawn': 8, 'lost': 9, 'goals_for': 32, 'goals_against': 35,
             'goal_difference': -3, 'points': 32, 'form': 'DDLWD'},
            {'position': 11, 'team_id': 402, 'team_name': 'Brentford', 'short_name': 'BRE',
             'played': 25, 'won': 8, 'drawn': 6, 'lost': 11, 'goals_for': 36, 'goals_against': 40,
             'goal_difference': -4, 'points': 30, 'form': 'LDLWL'},
            {'position': 12, 'team_id': 394, 'team_name': 'Wolverhampton', 'short_name': 'WOL',
             'played': 25, 'won': 7, 'drawn': 8, 'lost': 10, 'goals_for': 30, 'goals_against': 38,
             'goal_difference': -8, 'points': 29, 'form': 'WDLDL'},
            {'position': 13, 'team_id': 349, 'team_name': 'Ipswich Town', 'short_name': 'IPS',
             'played': 25, 'won': 7, 'drawn': 6, 'lost': 12, 'goals_for': 28, 'goals_against': 42,
             'goal_difference': -14, 'points': 27, 'form': 'LLWDL'},
            {'position': 14, 'team_id': 76, 'team_name': 'West Bromwich Albion', 'short_name': 'WBA',
             'played': 25, 'won': 6, 'drawn': 8, 'lost': 11, 'goals_for': 26, 'goals_against': 40,
             'goal_difference': -14, 'points': 26, 'form': 'DLLWD'},
            {'position': 15, 'team_id': 356, 'team_name': 'Sheffield United', 'short_name': 'SHE',
             'played': 25, 'won': 6, 'drawn': 7, 'lost': 12, 'goals_for': 24, 'goals_against': 42,
             'goal_difference': -18, 'points': 25, 'form': 'WLDLL'},
            {'position': 16, 'team_id': 340, 'team_name': 'Southampton', 'short_name': 'SOU',
             'played': 25, 'won': 6, 'drawn': 5, 'lost': 14, 'goals_for': 22, 'goals_against': 45,
             'goal_difference': -23, 'points': 23, 'form': 'LLWDL'},
            {'position': 17, 'team_id': 563, 'team_name': 'AFC Bournemouth', 'short_name': 'BOU',
             'played': 25, 'won': 5, 'drawn': 7, 'lost': 13, 'goals_for': 24, 'goals_against': 48,
             'goal_difference': -24, 'points': 22, 'form': 'DLLLD'},
            {'position': 18, 'team_id': 68, 'team_name': 'Norwich City', 'short_name': 'NOR',
             'played': 25, 'won': 4, 'drawn': 6, 'lost': 15, 'goals_for': 20, 'goals_against': 50,
             'goal_difference': -30, 'points': 18, 'form': 'LLDLL'},
        ])

        # Generate sample historical matches
        historical_matches = self._generate_sample_matches(teams_data)

        # Generate sample remaining fixtures
        remaining_fixtures = self._generate_sample_fixtures(teams_data)

        return standings, historical_matches, remaining_fixtures

    def _generate_sample_matches(self, teams_data: Dict) -> pd.DataFrame:
        """Generate sample historical matches."""
        import numpy as np
        np.random.seed(42)

        team_ids = teams_data['team_id'][:20]
        team_names = {tid: name for tid, name in zip(teams_data['team_id'], teams_data['team_name'])}

        matches = []
        match_id = 1000

        for season in config.TRAINING_SEASONS:
            # Generate matches for each season
            for matchday in range(1, 39):  # 38 matchdays
                # Each matchday has 10 matches
                teams_shuffled = np.random.permutation(team_ids)
                for i in range(0, len(teams_shuffled) - 1, 2):
                    if i + 1 < len(teams_shuffled):
                        home_id = int(teams_shuffled[i])
                        away_id = int(teams_shuffled[i + 1])

                        # Simulate realistic scores with home advantage
                        home_goals = np.random.poisson(1.5)
                        away_goals = np.random.poisson(1.1)

                        # Determine winner
                        if home_goals > away_goals:
                            winner = 'HOME_TEAM'
                        elif away_goals > home_goals:
                            winner = 'AWAY_TEAM'
                        else:
                            winner = 'DRAW'

                        matches.append({
                            'match_id': match_id,
                            'season': season,
                            'date': f"{season}-08-{15 + (matchday % 15):02d}T15:00:00Z",
                            'matchday': matchday,
                            'home_team_id': home_id,
                            'home_team': team_names.get(home_id, 'Unknown'),
                            'away_team_id': away_id,
                            'away_team': team_names.get(away_id, 'Unknown'),
                            'home_goals': home_goals,
                            'away_goals': away_goals,
                            'winner': winner,
                            'status': 'FINISHED'
                        })
                        match_id += 1

        return pd.DataFrame(matches)

    def _generate_sample_fixtures(self, teams_data: Dict) -> pd.DataFrame:
        """Generate sample remaining fixtures."""
        import numpy as np
        np.random.seed(42)

        team_ids = teams_data['team_id'][:20]
        team_names = {tid: name for tid, name in zip(teams_data['team_id'], teams_data['team_name'])}

        fixtures = []
        match_id = 5000

        # Generate remaining matches (13 matchdays * 10 matches = 130 matches)
        for matchday in range(26, 39):
            teams_shuffled = np.random.permutation(team_ids)
            for i in range(0, len(teams_shuffled) - 1, 2):
                if i + 1 < len(teams_shuffled):
                    home_id = int(teams_shuffled[i])
                    away_id = int(teams_shuffled[i + 1])

                    fixtures.append({
                        'match_id': match_id,
                        'season': config.CURRENT_SEASON,
                        'date': f"2025-02-{15 + (matchday % 14):02d}T15:00:00Z",
                        'matchday': matchday,
                        'home_team_id': home_id,
                        'home_team': team_names.get(home_id, 'Unknown'),
                        'away_team_id': away_id,
                        'away_team': team_names.get(away_id, 'Unknown'),
                        'status': 'SCHEDULED'
                    })
                    match_id += 1

        return pd.DataFrame(fixtures)


def main():
    """Test data fetching."""
    fetcher = DataFetcher()

    # Try to get real data, fall back to sample data
    print("Attempting to fetch real data...")
    standings = fetcher.fetch_current_standings(use_cache=True)

    if standings.empty:
        print("Using sample data...")
        standings, historical, fixtures = fetcher.load_or_generate_sample_data()
        print(f"Standings shape: {standings.shape}")
        print(f"Historical matches shape: {historical.shape}")
        print(f"Remaining fixtures shape: {fixtures.shape}")

        # Save sample data
        standings.to_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv", index=False)
        historical.to_csv(f"{config.DATA_PROCESSED_DIR}/historical_matches.csv", index=False)
        fixtures.to_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv", index=False)
    else:
        print("Real data fetched successfully!")
        print(standings.head())

        # Save
        standings.to_csv(f"{config.DATA_PROCESSED_DIR}/current_standings.csv", index=False)

        # Fetch historical and fixtures
        historical = fetcher.fetch_historical_data()
        fixtures = fetcher.fetch_remaining_fixtures()

        historical.to_csv(f"{config.DATA_PROCESSED_DIR}/historical_matches.csv", index=False)
        fixtures.to_csv(f"{config.DATA_PROCESSED_DIR}/remaining_fixtures.csv", index=False)


if __name__ == '__main__':
    main()
