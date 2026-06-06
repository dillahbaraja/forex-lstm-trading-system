# Forex LSTM Trading System

Forex trading system based on LSTM with MQL5 integration, ONNX deployment, training pipeline, and backtesting for price prediction experiments and trading environment validation.

## Contents

- Python scripts for data preparation, training, and reference collection
- MQL5 source files and runtime configuration
- ONNX model artifacts and scaler/feature contracts
- Supporting scripts, workflow notes, and benchmark assets

## Excluded

- Research article drafts and manuscript files
- Large raw training outputs and temporary checkpoint folders
- Disposable runtime caches and compiled binaries

## Main Components

- `Python_Scripts/`
- `MQL5/`
- `MQL5_EA/`
- `GAS_Scripts/`
- `Run_Training.bat`
- `Run_Benchmark.bat`
- `TradingBot.mq5`
- `DataExporter.mq5`
- `model_*.onnx`
- `feature_contract.json`
- `label_config.json`
- `scaler_params.csv`

## Usage

### 1) Data Collection

- Export market data from MT5 using `DataExporter.mq5` if you need fresh CSV data.
- Place the exported CSV files in the project root or inside `Python_Scripts/` with the expected naming convention, such as `EURUSD_H1_Data.csv` or `USDJPY_H1_Data.csv`.
- If you are using the existing data files, you can skip this step.

### 2) Data Preparation

- Run `Python_Scripts/data_preprocessing.py` to clean and transform the raw CSV data.
- This step prepares the input features, labels, and scaling needed for training.
- Supporting metadata such as `feature_contract.json`, `label_config.json`, and `scaler_params.csv` is used by the pipeline and MT5 deployment flow.

### 3) Model Training

- Run `Run_Training.bat` to start the training pipeline.
- The main training logic is in `Python_Scripts/train_model.py`.
- Training outputs include model artifacts such as `.onnx` files, scaler parameters, and report files.

### 4) Testing / Backtesting

- Use `TradingBot.mq5` in MetaTrader 5 to test the strategy logic with the generated model files.
- For benchmark-style checks, you can also run `Run_Benchmark.bat`.
- Model files are expected to be available in the MT5 working paths under `MQL5/Files/` or `MQL5_EA/`, depending on the deployment setup.

### 5) Evaluation

- Review `training_report.json` and the pair-specific reports such as `training_report_eurjpy_H1.json` and `training_report_usdjpy_H1.json`.
- Use the benchmark outputs and generated charts to compare model behavior across runs.
- Compare backtest results with the training reports to judge whether the model is stable enough for further experimentation.

## Notes

The repository is organized as a backup and working copy for experimentation, model export, and MT5 deployment.
