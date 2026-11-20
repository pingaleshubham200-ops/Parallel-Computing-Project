# Smart Grid Real-Time Load Monitoring Dataset

This directory contains the dataset used for parallel electricity consumption and peak load analysis.

## 1. Primary Dataset: Kaggle Smart Grid Real-Time Load Monitoring
- **Kaggle Identifier**: `ziya07/smart-grid-real-time-load-monitoring-dataset`
- **File**: `smart_grid_dataset.csv`
- **Total Records**: 50,000 real-time measurements recorded across ~1.5 years (from `2024-01-01 00:00:00` to `2025-06-04 19:45:00`).
- **Sampling Interval**: 15 minutes ($\Delta t = 0.25$ hours).
- **File Size**: ~12.6 MB.

### Automated Download via `kagglehub`
You can download and extract this dataset directly using Python and `kagglehub`:
```bash
python3 scripts/download_kaggle_dataset.py
```
Or via Makefile:
```bash
make download-kaggle
```

Or directly in Python:
```python
import kagglehub
path = kagglehub.dataset_download("ziya07/smart-grid-real-time-load-monitoring-dataset")
print("Path to dataset files:", path)
```

## 2. Schema and Attribute Definitions

| Column | Unit / Format | Data Type | Description |
|:---|:---|:---|:---|
| `Timestamp` | `YYYY-MM-DD HH:MM:SS` | String | Measurement timestamp sampled at 15-minute intervals |
| `Voltage (V)` | Volts (V) | Float | Line voltage delivered by the substation (nominal 230 V) |
| `Current (A)` | Amperes (A) | Float | Aggregate electrical load current |
| `Power Consumption (kW)` | **Kilowatts (kW)** | Float | **Primary active load demand metric** |
| `Reactive Power (kVAR)` | kVAR | Float | Reactive component of grid power |
| `Power Factor` | Ratio ($0-1$) | Float | $\cos\phi$ real-to-apparent power efficiency |
| `Solar Power (kW)` | kW | Float | Renewable generation from localized photovoltaic arrays |
| `Wind Power (kW)` | kW | Float | Renewable generation from wind turbines |
| `Grid Supply (kW)` | kW | Float | External transmission grid feed |
| `Voltage Fluctuation (%)` | % | Float | Percentage deviation from nominal voltage rating |
| `Overload Condition` | Binary ($0/1$) | Integer | Grid overload alert flag ($1 = \text{overload active}$) |
| `Transformer Fault` | Binary ($0/1$) | Integer | Substation transformer fault alert flag ($1 = \text{fault}$) |
| `Temperature (°C)` | °C | Float | Ambient temperature |
| `Humidity (%)` | % | Float | Ambient relative humidity |
| `Electricity Price (USD/kWh)` | USD / kWh | Float | Time-of-use dynamic electricity tariff |
| `Predicted Load (kW)` | kW | Float | Baseline forecasted demand from ML load model |

## 3. Power and Energy Calculations
Since telemetry readings are logged every 15 minutes ($\Delta t = 15/60 = 0.25$ hours):
- **Cumulative Electrical Energy**:
  $$E = \sum_{i=1}^{N} P_i \times 0.25 \quad (\text{kWh})$$
- **Total Renewable Energy**:
  $$E_{\text{renewable}} = \sum_{i=1}^{N} (P_{\text{solar}, i} + P_{\text{wind}, i}) \times 0.25 \quad (\text{kWh})$$
- **Total Electricity Cost**:
  $$\text{Cost} = \sum_{i=1}^{N} (P_i \times 0.25 \times \text{Price}_i) \quad (\text{USD})$$
