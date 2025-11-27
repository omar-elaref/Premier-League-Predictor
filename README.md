# Premier League Predictor

## Overview

This project implements a neural network-based approach to predict football match scores in the English Premier League. The model leverages team form, head-to-head history, betting odds, and team embeddings to make predictions.

## Features

- **Sequence-based modeling**: Uses GRU encoders to process team form and head-to-head history
- **Multi-feature integration**: Combines team embeddings, betting odds, and historical match sequences
- **Time-based validation**: Uses the most recent season for validation while training on historical data
- **Comprehensive metrics**: Tracks exact score accuracy, win/draw/loss accuracy, and RMSE
- **League table generation**: Produces predicted league tables based on model predictions

## Architecture

The model (`FootballScorePredictor`) consists of:

1. **Team Embeddings**: Learnable embeddings for each team
2. **Match History Encoders**: GRU-based encoders for:
   - Head-to-head history between teams
   - Recent form sequences for each team
3. **Metadata Processing**: MLP to process team embeddings, ground flags, and betting odds
4. **Final Predictor**: Feedforward network that combines all features to predict goals

### Key Components

- `MatchHistoryEncoder`: GRU-based encoder for processing match sequences
- `FootballScorePredictor`: Main model architecture
- `FootballSequenceDataset`: PyTorch dataset for match data

## Installation

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- scikit-learn

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Premier-League-Predictor

# Install dependencies
pip install torch numpy pandas scikit-learn
```

## Data

The project expects Premier League match data in CSV format in the `Premier League/` directory. Each CSV file should contain match data for a season (e.g., `2010-11.csv`, `2011-12.csv`, etc.).

Required columns include:

- `Date`, `HomeTeam`, `AwayTeam`
- `FTHG`, `FTAG`, `FTR` (full-time goals and result)
- `B365H`, `B365D`, `B365A` (betting odds)

## Usage

### Training the Model

Run the main training script:

```bash
python prem_learning.py
```

This will:

1. Load all Premier League seasons from CSV files
2. Build the dataset with team form and head-to-head sequences
3. Split data temporally (last season for validation)
4. Train the model for 30 epochs (This can be updated in the `prem_learning.py` file)
5. Display training metrics and generate a predicted league table

### Model Configuration

Key hyperparameters in `prem_learning.py`:

- `k_form=5`: Number of recent matches for team form
- `k_h2h=5`: Number of recent head-to-head matches
- `batch_size=64`: Training batch size
- `num_epochs=30`: Number of training epochs
- `learning_rate=1e-3`: Adam optimizer learning rate

## Project Structure

```
Premier-League-Predictor/
├── architecture.py          # Model architecture definitions
├── build_dataset.py         # Dataset construction and feature engineering
├── importing_files.py       # Data loading utilities
├── prem_learning.py         # Main training script
├── print_results.py        # Validation table generation
├── Premier League/         # Match data CSV files
└── docs/                   # Documentation and reports
```

## Model Details

### Input Features

1. **Team IDs**: Categorical team identifiers
2. **Ground Flags**: Home/away indicator
3. **Betting Odds**: B365H, B365D, B365A (Bet365 odds)
4. **Team Form Sequences**: Last k_form matches for each team
5. **Head-to-Head Sequences**: Last k_h2h matches between the two teams

### Match Features

Each match in a sequence is represented by 7 features:

- Goals for (GF)
- Goals against (GA)
- Goal difference (GD)
- Home/away flag
- Win/Draw/Loss indicators

### Output

The model predicts a 2D vector `[goals_team1, goals_team2]` representing the predicted scoreline.

### Metrics

- **RMSE**: Root mean squared error on goal predictions
- **Exact Score Accuracy**: Percentage of matches with exact scoreline predictions
- **W/D/L Accuracy**: Percentage of matches with correct result (win/draw/loss)

## Results

After training, the model outputs:

- Training and validation loss per epoch
- Validation metrics (RMSE, exact score accuracy, W/D/L accuracy)
- Predicted league table for the validation season
