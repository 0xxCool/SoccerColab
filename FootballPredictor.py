import os
import multiprocessing
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import requests
import lightgbm as lgbm
import warnings
import tensorflow as tf
import optuna
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor, Pool
import joblib
from datetime import datetime, timezone, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import yaml
import unittest
from functools import partial
import asyncio
import aiohttp

# Konfigurieren von Warnungen und Logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Laden der Konfiguration
def load_config() -> Dict[str, Any]:
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()
API_KEY = config['api_key']
N_CORES = multiprocessing.cpu_count()

# Asynchrone Funktion zum Abrufen von Daten
async def fetch_competition_data(session: aiohttp.ClientSession, competition: str) -> List[Dict[str, Any]]:
    url = f"http://api.football-data.org/v4/competitions/{competition}/matches"
    headers = {'X-Auth-Token': API_KEY}
    
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            return [{
                'Date': datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ'),
                'HomeTeam': str(match['homeTeam']['name']),
                'AwayTeam': str(match['awayTeam']['name']),
                'HomeScore': match['score']['fullTime']['home'],
                'AwayScore': match['score']['fullTime']['away'],
                'Competition': competition
            } for match in data['matches']]
        else:
            logging.error(f"Error fetching data for competition {competition}: {response.status}")
            return []

async def fetch_data_from_api() -> pd.DataFrame:
    all_matches = []
    competitions = config['competitions']

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_competition_data(session, competition) for competition in competitions]
        results = await asyncio.gather(*tasks)
        
        for matches in results:
            all_matches.extend(matches)
    
    if not all_matches:
        logging.warning("No matches fetched. Check your API key and network connection.")
        return pd.DataFrame()
    
    return pd.DataFrame(all_matches)

# Optimiertes Feature-Engineering
def advanced_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    
    # Vektorisierte Operationen
    data['TotalGoals'] = data['HomeScore'].fillna(0) + data['AwayScore'].fillna(0)
    data['GoalDifference'] = data['HomeScore'].fillna(0) - data['AwayScore'].fillna(0)

    # Team-Form (vektorisiert)
    for team_type in ['Home', 'Away']:
        data[f'{team_type}Form'] = data.groupby(f'{team_type}Team')['GoalDifference'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    
    # Head-to-Head Statistiken
    h2h_stats = data.groupby(['HomeTeam', 'AwayTeam']).agg({
        'HomeScore': 'mean',
        'AwayScore': 'mean'
    }).reset_index()
    h2h_stats.columns = ['HomeTeam', 'AwayTeam', 'H2H_HomeGoals', 'H2H_AwayGoals']
    data = pd.merge(data, h2h_stats, on=['HomeTeam', 'AwayTeam'], how='left')
    
    # Liga-Position und Tordifferenz (vektorisiert)
    for team_type in ['Home', 'Away']:
        data[f'{team_type}LeaguePosition'] = data.groupby('Date')[f'{team_type}Team'].rank(method='dense')
        data[f'{team_type}GoalDifference'] = data.groupby(f'{team_type}Team')['GoalDifference'].cumsum()
    
    # Weitere abgeleitete Features
    data['HomeAwayRatio'] = data['HomeForm'] / (data['AwayForm'] + 1e-5)
    data['FormMomentum'] = data['HomeForm'].diff() - data['AwayForm'].diff()
    data['SeasonProgress'] = data.groupby('Competition')['Date'].rank(method='dense') / data.groupby('Competition')['Date'].transform('count')
    
    # Zeitbasierte Features
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['MonthOfYear'] = data['Date'].dt.month

    # Zusätzliche Features
    data['RecentFormHome'] = data.groupby('HomeTeam')['GoalDifference'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    data['RecentFormAway'] = data.groupby('AwayTeam')['GoalDifference'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    data['HomeTeamStrength'] = data.groupby('HomeTeam')['GoalDifference'].transform('cumsum') / (data.groupby('HomeTeam').cumcount() + 1)
    data['AwayTeamStrength'] = data.groupby('AwayTeam')['GoalDifference'].transform('cumsum') / (data.groupby('AwayTeam').cumcount() + 1)

    # NaN-Werte behandeln
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    return data

# Modelltraining und -vorhersage
def create_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = [
        'HomeForm', 'AwayForm', 'H2H_HomeGoals', 'H2H_AwayGoals',
        'HomeLeaguePosition', 'AwayLeaguePosition', 'HomeGoalDifference', 'AwayGoalDifference',
        'HomeAwayRatio', 'FormMomentum', 'SeasonProgress',
        'DayOfWeek', 'IsWeekend', 'MonthOfYear',
        'RecentFormHome', 'RecentFormAway', 'HomeTeamStrength', 'AwayTeamStrength'
    ]
    X = data[features]
    y_home = data['HomeScore']
    y_away = data['AwayScore']
    return X, y_home, y_away

# Optimiertes Modelltraining mit Hyperparameter-Tuning
def optimize_xgboost(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    params = {
        'n_jobs': -1,
        'gpu_id': 0,
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'tree_method': 'gpu_hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    return params

def optimize_lightgbm(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_thread': N_CORES,
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 0.1, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'verbose': -1
    }
    return params

def optimize_catboost(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    params = {
        'task_type': 'GPU',
        'devices': '0:1',
        'thread_count': N_CORES,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    return params

def train_models(X: np.ndarray, y_home: np.ndarray, y_away: np.ndarray, gpu_available: bool = False) -> Dict[str, Any]:
    def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, model_type: str) -> float:
        if model_type == 'xgboost':
            params = optimize_xgboost(trial, X, y)
            model = XGBRegressor(**params)
        elif model_type == 'lightgbm':
            params = optimize_lightgbm(trial, X, y)
            model = LGBMRegressor(**params)
        else:  # catboost
            params = optimize_catboost(trial, X, y)
            model = CatBoostRegressor(**params)
        
        scores = []
        kf = TimeSeriesSplit(n_splits=5)
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            if model_type == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
                )
            elif model_type == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            else:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            
            preds = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(score)
        
        return np.mean(scores)

    models = {}
    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X, y_home if model_type != 'catboost' else y_away, model_type), n_trials=50)
        best_params = study.best_params
        
        if model_type == 'xgboost':
            models[model_type] = XGBRegressor(**best_params, tree_method="gpu_hist" if gpu_available else "hist")
        elif model_type == 'lightgbm':
            models[model_type] = LGBMRegressor(**best_params, device='gpu' if gpu_available else 'cpu')
        else:  # catboost
            models[model_type] = CatBoostRegressor(**best_params, task_type='GPU' if gpu_available else 'CPU')
        
        models[model_type].fit(X, y_home if model_type != 'catboost' else y_away)

    return models

def ensemble_predict(models: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    predictions = []
    for model_name, model in models.items():
        try:
            pred = model.predict(X)
            predictions.append(pred)
            logging.info(f"{model_name} prediction successful")
        except Exception as e:
            logging.error(f"Error in {model_name} prediction: {str(e)}")
            predictions.append(np.zeros(len(X)))

    home_pred = np.mean([predictions[0], predictions[1]], axis=0)
    away_pred = predictions[2]

    return home_pred, away_pred

# Over/Under Vorhersage
def predict_over_under(home_goals: float, away_goals: float) -> str:
    total_goals = home_goals + away_goals
    return 'Over' if total_goals > 2.5 else 'Under'

# Verbesserte GUI
class FootballPredictionsApp:
    def __init__(self, master: tk.Tk, models: Dict[str, Any], data: pd.DataFrame):
        self.master = master
        self.master.title("Football Match Predictions ⚽ Over/Under 2.5 Goals")
        self.master.geometry("1200x600")
        self.master.configure(bg='#f0f0f0')  # Hintergrundfarbe

        self.models = models
        self.data = data
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        self.create_widgets()

    def create_widgets(self):
        # Stil für Widgets
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', background='#4CAF50', foreground='white', font=('Arial', 10, 'bold'))
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TEntry', font=('Arial', 10))

        # Hauptframe
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Titelleiste
        title_label = ttk.Label(main_frame, text="Football Match Predictor ⚽ Over/Under 2.5 Goals", 
                                font=("Arial", 16, "bold"), background='#4CAF50', foreground='white')
        title_label.pack(fill=tk.X, pady=(0, 10))

        # Suchleiste
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(search_frame, text="Search Team:").pack(side=tk.LEFT, padx=(0, 5))
        self.team_search_var = tk.StringVar()
        self.team_search_entry = ttk.Entry(search_frame, textvariable=self.team_search_var, width=30)
        self.team_search_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.team_search_entry.bind("<KeyRelease>", self.filter_treeview)

        ttk.Label(search_frame, text="Search Date:").pack(side=tk.LEFT, padx=(0, 5))
        self.cal = DateEntry(search_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.cal.pack(side=tk.LEFT, padx=(0, 5))
        self.cal.set_date(self.data['Date'].max().date())

        search_button = ttk.Button(search_frame, text="Search", command=self.search_date)
        search_button.pack(side=tk.LEFT)

        # Treeview
        self.tree_frame = ttk.Frame(main_frame)
        self.tree_frame.pack(expand=True, fill=tk.BOTH, pady=10)

        columns = ('Date', 'HomeTeam', 'AwayTeam', 'Predicted Home Goals', 'Predicted Away Goals', 'Over/Under (2.5)')
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show='headings')
        self.tree.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        for col in columns:
            self.tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(_col, False))
            self.tree.column(col, anchor="center", width=120)

        scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.populate_treeview(self.data)

    def populate_treeview(self, data: pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        for _, row in data.iterrows():
            self.tree.insert("", "end", values=(
                row['Date'].strftime('%Y-%m-%d %H:%M'),
                row['HomeTeam'],
                row['AwayTeam'],
                f"{row['Predicted Home Goals']:.2f}",
                f"{row['Predicted Away Goals']:.2f}",
                row['Over/Under (2.5)']
            ))

    def filter_treeview(self, event=None):
        search_term = self.team_search_var.get().lower()
        filtered_data = self.data[self.data.apply(lambda row: 
            search_term in row['HomeTeam'].lower() or search_term in row['AwayTeam'].lower(), axis=1)]
        self.populate_treeview(filtered_data)

    def search_date(self):
        selected_date = self.cal.get_date()
        start_date = datetime.combine(selected_date, datetime.min.time())
        end_date = start_date + timedelta(days=1)
        filtered_data = self.data[(self.data['Date'] >= start_date) & (self.data['Date'] < end_date)]
        if filtered_data.empty:
            messagebox.showinfo("No Matches", f"No matches found for {selected_date}")
        else:
            self.populate_treeview(filtered_data)

    def treeview_sort_column(self, col, reverse):
        l = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        l.sort(key=lambda t: t[0].lower(), reverse=reverse)

        for index, (val, k) in enumerate(l):
            self.tree.move(k, '', index)

        self.tree.heading(col, command=lambda: self.treeview_sort_column(col, not reverse))

def main():
    logging.info("Starting the Football Predictor application")

    try:
        # GPU-Konfiguration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU is available. Found {len(gpus)} GPU(s)")
                gpu_available = True
            except RuntimeError as e:
                logging.error(f"Error configuring GPU: {e}")
                gpu_available = False
        else:
            logging.info("No GPU available. Using CPU.")
            gpu_available = False

        logging.info("Fetching data from API...")
        data = asyncio.run(fetch_data_from_api())
        if data.empty:
            logging.error("No data fetched from API. Exiting program.")
            return

        logging.info("Performing advanced feature engineering...")
        data = advanced_feature_engineering(data)

        logging.info("Creating features and targets...")
        X, y_home, y_away = create_features_target(data)

        logging.info("Scaling and imputing data...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X_scaled)

        # Ask user whether to train new models or load existing ones
        user_choice = input("Do you want to train new models or load existing ones? (train/load): ").lower()
        
        models_file = 'models.joblib'
        if user_choice == 'train':
            logging.info("Training new models...")
            models = train_models(X_imputed, y_home, y_away, gpu_available=gpu_available)
            joblib.dump(models, models_file)
            logging.info(f"Models saved to {models_file}")
        elif user_choice == 'load':
            logging.info("Loading existing models...")
            try:
                models = joblib.load(models_file)
                logging.info(f"Models loaded from {models_file}")
            except FileNotFoundError:
                logging.info("No existing models found. Training new models...")
                models = train_models(X_imputed, y_home, y_away, gpu_available=gpu_available)
                joblib.dump(models, models_file)
                logging.info(f"Models saved to {models_file}")
        else:
            logging.error("Invalid choice. Exiting program.")
            return

        logging.info("Making predictions...")
        home_pred, away_pred = ensemble_predict(models, X_imputed)
        
        data['Predicted Home Goals'] = home_pred
        data['Predicted Away Goals'] = away_pred
        data['Over/Under (2.5)'] = data.apply(lambda row: predict_over_under(row['Predicted Home Goals'], row['Predicted Away Goals']), axis=1)

        # Calculate and display prediction accuracy
        actual_total_goals = data['HomeScore'] + data['AwayScore']
        predicted_total_goals = data['Predicted Home Goals'] + data['Predicted Away Goals']
        correct_predictions = ((actual_total_goals > 2.5) == (predicted_total_goals > 2.5)).sum()
        accuracy = (correct_predictions / len(data)) * 100
        logging.info(f"Prediction Accuracy: {accuracy:.2f}%")

        # Evaluate using MSE and R2 score
        mse_home = mean_squared_error(y_home, home_pred)
        mse_away = mean_squared_error(y_away, away_pred)
        r2_home = r2_score(y_home, home_pred)
        r2_away = r2_score(y_away, away_pred)
        logging.info(f"MSE Home: {mse_home:.4f}, MSE Away: {mse_away:.4f}")
        logging.info(f"R2 Score Home: {r2_home:.4f}, R2 Score Away: {r2_away:.4f}")

        logging.info("Launching GUI...")
        root = tk.Tk()
        app = FootballPredictionsApp(root, models, data)
        root.mainloop()

    except Exception as e:
        logging.error(f"An error occurred in main(): {str(e)}")
        raise

# Erweiterte Testfälle
class TestFootballPredictor(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'HomeTeam': ['Team A'] * 10,
            'AwayTeam': ['Team B'] * 10,
            'HomeScore': np.random.randint(0, 5, 10),
            'AwayScore': np.random.randint(0, 5, 10),
            'Competition': ['PL'] * 10
        })

    def test_advanced_feature_engineering(self):
        processed_data = advanced_feature_engineering(self.sample_data)
        self.assertGreater(len(processed_data.columns), len(self.sample_data.columns))
        self.assertIn('TotalGoals', processed_data.columns)
        self.assertIn('GoalDifference', processed_data.columns)
        self.assertIn('RecentFormHome', processed_data.columns)
        self.assertIn('RecentFormAway', processed_data.columns)

    def test_create_features_target(self):
        processed_data = advanced_feature_engineering(self.sample_data)
        X, y_home, y_away = create_features_target(processed_data)
        self.assertEqual(len(X), len(self.sample_data))
        self.assertEqual(len(y_home), len(self.sample_data))
        self.assertEqual(len(y_away), len(self.sample_data))

    def test_predict_over_under(self):
        self.assertEqual(predict_over_under(2, 1), 'Over')
        self.assertEqual(predict_over_under(1, 1), 'Under')
        self.assertEqual(predict_over_under(0, 2), 'Under')

    def test_ensemble_predict(self):
        # Create dummy models and data for testing
        class DummyModel:
            def predict(self, X):
                return np.random.rand(len(X))

        dummy_models = {
            'xgboost': DummyModel(),
            'lightgbm': DummyModel(),
            'catboost': DummyModel()
        }
        dummy_X = np.random.rand(100, 10)

        home_pred, away_pred = ensemble_predict(dummy_models, dummy_X)
        self.assertEqual(len(home_pred), 100)
        self.assertEqual(len(away_pred), 100)

    def test_train_models(self):
        X = np.random.rand(100, 18)  # 18 features as in create_features_target
        y_home = np.random.rand(100)
        y_away = np.random.rand(100)
        models = train_models(X, y_home, y_away, gpu_available=False)
        self.assertIn('xgboost', models)
        self.assertIn('lightgbm', models)
        self.assertIn('catboost', models)

    @unittest.mock.patch('aiohttp.ClientSession.get')
    async def test_fetch_competition_data(self, mock_get):
        # Mock the API response
        mock_response = unittest.mock.Mock()
        mock_response.status = 200
        mock_response.json.return_value = {
            'matches': [
                {
                    'utcDate': '2023-01-01T15:00:00Z',
                    'homeTeam': {'name': 'Team A'},
                    'awayTeam': {'name': 'Team B'},
                    'score': {'fullTime': {'home': 2, 'away': 1}}
                }
            ]
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        async with aiohttp.ClientSession() as session:
            result = await fetch_competition_data(session, 'PL')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['HomeTeam'], 'Team A')
        self.assertEqual(result[0]['AwayTeam'], 'Team B')
        self.assertEqual(result[0]['HomeScore'], 2)
        self.assertEqual(result[0]['AwayScore'], 1)

if __name__ == "__main__":
    unittest.main(exit=False)
    main()
