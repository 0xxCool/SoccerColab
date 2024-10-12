# Football Predictor: Over/Under 2.5 Goals ⚽

## Overview

Football Predictor is an advanced machine learning application that predicts whether a football (soccer) match will have over or under 2.5 goals. It uses data from various top European leagues, employs ensemble learning techniques, and provides a user-friendly GUI for interaction.

![Football Predictor Screenshot](screenshot.png)

## Features

- **Data Fetching**: Asynchronously fetches match data from the football-data.org API.
- **Advanced Feature Engineering**: Creates complex features from raw match data.
- **Ensemble Learning**: Utilizes XGBoost, LightGBM, and CatBoost for predictions.
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning.
- **GPU Acceleration**: Supports GPU for faster model training and prediction.
- **User-Friendly GUI**: Provides an intuitive interface for viewing predictions and filtering results.
- **Flexible Configuration**: Easily customizable through a YAML configuration file.

## Requirements

- Python 3.8+
- See `requirements.txt` for a full list of Python package dependencies.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/football-predictor.git
   cd football-predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your configuration:
   - Copy `config.yaml.example` to `config.yaml`
   - Edit `config.yaml` and add your football-data.org API key

## Usage

1. Run the main script:
   ```
   python football_predictor.py
   ```

2. The GUI will launch, allowing you to:
   - View predictions for upcoming matches
   - Filter matches by team or date
   - Sort results by different criteria

3. The program will ask if you want to train new models or use existing ones. For first-time use, select 'train'.

## Configuration

You can customize various aspects of the predictor by editing `config.yaml`:

- API key
- Leagues to fetch data for
- Model parameters
- GUI settings
- Logging configuration

See `config.yaml.example` for a full list of configurable options.

## Development

- Use `black` for code formatting: `black .`
- Run tests with: `python -m unittest discover tests`
- Check type hints with mypy: `mypy football_predictor.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data provided by [football-data.org](https://www.football-data.org/)
- Inspired by various sports prediction models and academic papers on the subject.

## Disclaimer

This tool is for educational and entertainment purposes only. Please gamble responsibly and be aware of the risks involved in sports betting.
