# Ethanol-Water ASNN (PyTorch) VLE Prediction

This repository implements a structured artificial neural network (ASNN) for vapor-liquid equilibrium (VLE) prediction of ethanol-water binary mixtures. The model uses a thermodynamically inspired architecture similar to NRTL and was converted to PyTorch for CPU compatibility.

## Project Structure

- `components.csv`: Component property metadata.
- `system_index.csv`: Mixture system indexing.
- `vle_dataset_clean.csv`: Cleaned experimental VLE dataset.
- `ethanol_water_asnn.ipynb`: Main notebook with full pipeline (data load → model → evaluation → save outputs).
- `README.md`: This file.

## Problems Encountered

1. TensorFlow not installing correctly on CPU environment.
2. Initial code used Keras idioms (`model.save`, `history.history`) incompatible with PyTorch.
3. Need to update metrics, model persistence, and training pipeline to PyTorch best practices.

## Methodology

### Data processing
- Filter ethanol-water rows (either order of components).
- Define features: `x1`, `T_K`, `Psat1_kPa`, `Psat2_kPa`.
- Define targets: `P_kPa`, `y1`.
- Train/test/val split: every 5th point reserved for holdout; half of those used for validation, half for test.
- Standard scaling fitted on training data and applied to val/test.

### Model Architecture (ASNN)
- PyTorch `nn.Module` with sequential layers:
  - Dense blocks + `LayerNorm` + `Tanh` + `Dropout`
  - Structured pathway to capture nonlinear VLE behavior.
- Output dimension: 2 (pressure and vapor composition).

### Training
- Loss: `MSELoss`.
- Optimizer: Adam.
- Early stopping with patience.
- Epochs around 200 (or until early stop).

### Evaluation
- Metrics: RMSE, MAE, R², AARD (Average Absolute Relative Deviation) for test set.
- Visualization: Loss curves, predictions vs actual, residuals.
- Save artifacts: model state dict, scalers, metrics, results summary, predictions, plots.

## Result and Discussion

- Model converges and generalizes under structured split.
- AARD and R² help quantify predictive quality for both `P_kPa` and `y1`.
- Caution: test subset is selected systematically (every 5th row), preserving broad coverage but still depends on dataset ordering.
- Recommendation: perform cross-validation across samples and per-temperature strata for robustness.

## Output files
- Model: `asnn_ethanol_water_model.pth`
- Scalers: `feature_scaler.pkl`, `target_scaler.pkl`
- History: `training_history.pkl`
- Summary: `asnn_results_summary.json`, `asnn_results_summary.pkl`
- Predictions: `test_predictions.csv`
- Plots: `asnn_training_results.png`, `asnn_residual_plots.png`

## How to run

1. Install dependencies:
   - `pip install pandas numpy matplotlib seaborn scikit-learn torch`
2. Open `ethanol_water_asnn.ipynb` in Jupyter.
3. Run cells sequentially.
4. Check final summary and saved files.

## Notes

- The code is currently tuned for CPU. GPU support is optional by changing `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- For heavy dataset or larger networks, tune batch size and learning rate.

## Contact

- For enhancements or bug fixes, modify the notebook and rerun the full pipeline. 
