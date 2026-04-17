import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
import hashlib
import json
from datetime import datetime

# --------------------------------------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------------------------------------
df = pd.read_csv(
    "OBD_2024_I_2026-01-20T05_55_07.csv",
    sep=";",
    encoding="latin1",
    low_memory=False
)

material_uuid = "c93da4c3-94c9-4c86-b092-610cf1cf012f"
material_df = df[df["UUID"] == material_uuid]

features = [
    "Modul", "GWP", "AP", "EP", "ADPF", "ADPE",
    "PENRE", "PERE"
]

material_df = material_df[features]

# Replace invalid missing markers
material_df = material_df.replace(
    ["", " ", "NA", "N/A", "n.a.", "-", None],
    np.nan
)

# Convert numeric columns
numeric_cols = material_df.columns.drop("Modul")
material_df[numeric_cols] = material_df[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Fill missing values
material_df = material_df.fillna(0)

# Replace zero values with random numbers for specific columns
cols_to_randomize = ["GWP", "AP", "EP", "ADPF", "ADPE"]
for col in cols_to_randomize:
    mask = material_df[col] == 0
    material_df.loc[mask, col] = np.random.uniform(0.1, 100.0, size=mask.sum())

# Sort lifecycle order
stage_mapping = {
    "A1-A3": 1,
    "A4": 2,
    "A5": 3,
    "C1": 4,
    "C2": 5,
    "C3": 6,
    "C4": 7,
    "D": 8
}

material_df["stage_index"] = material_df["Modul"].map(stage_mapping)
material_df = material_df.sort_values("stage_index")

# Handle duplicate stages
duplicate_stages = material_df["Modul"].value_counts()
if (duplicate_stages > 1).any():
    material_df = material_df.groupby("Modul", as_index=False).agg({
        "GWP": "mean",
        "AP": "mean",
        "EP": "mean",
        "ADPF": "mean",
        "ADPE": "mean",
        "PENRE": "mean",
        "PERE": "mean",
        "stage_index": "first"
    })
    material_df = material_df.sort_values("stage_index")

# --------------------------------------------------------------------------------
# Create LSTM Sequences
# --------------------------------------------------------------------------------
n_samples = 1000
n_timesteps = len(material_df)
features_list = ["GWP", "AP", "EP", "ADPF", "ADPE"]
n_features = len(features_list)

X_synthetic = []
y_synthetic = []

base_matrix = material_df[features_list].values

np.random.seed(42)
for _ in range(n_samples):
    noise = np.random.normal(0, 0.1 * np.std(base_matrix, axis=0), base_matrix.shape)
    X_sample = base_matrix + noise
    total_gwp = np.sum(X_sample[:, 0])
    y_sample = total_gwp + np.random.normal(0, 0.1)
    X_synthetic.append(X_sample)
    y_synthetic.append(y_sample)

X_synthetic = np.array(X_synthetic)
y_synthetic = np.array(y_synthetic)

# Scale the features
X_reshaped = X_synthetic.reshape(-1, n_features)
scaler = MinMaxScaler()
X_scaled_flat = scaler.fit_transform(X_reshaped)
X_lstm = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)

y_lstm = y_synthetic.reshape(-1, 1)
y_scaler = MinMaxScaler()
y_lstm_scaled = y_scaler.fit_transform(y_lstm)

# --------------------------------------------------------------------------------
# Build and Train LSTM Model
# --------------------------------------------------------------------------------
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(n_timesteps, n_features)),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training LSTM Model...")
model.fit(X_lstm, y_lstm_scaled, epochs=800, verbose=0)
print("Model training complete!")

# --------------------------------------------------------------------------------
# Predictions and Evaluation
# --------------------------------------------------------------------------------
predicted_scaled = model.predict(X_lstm, verbose=0)
predicted_score = y_scaler.inverse_transform(predicted_scaled)

rmse = np.sqrt(mean_squared_error(y_lstm, predicted_score))
mae = mean_absolute_error(y_lstm, predicted_score)
r2 = r2_score(y_lstm, predicted_score)

print("\n" + "=" * 80)
print("MODEL EVALUATION METRICS")
print("=" * 80)
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R² Score (Coefficient of Determination): {r2:.4f}")
print("=" * 80)

# Calculate sustainability scores
material_df_reset = material_df.reset_index(drop=True)
indicators = ["GWP", "AP", "EP", "ADPF", "ADPE"]
normalized_impacts = pd.DataFrame()

for indicator in indicators:
    values = material_df_reset[indicator].values
    norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
    normalized_impacts[indicator] = norm_values

material_df_reset["composite_impact"] = normalized_impacts.mean(axis=1)
material_df_reset["sustainability_score"] = (1 - material_df_reset["composite_impact"]) * 100

best_stage = material_df_reset.loc[material_df_reset["sustainability_score"].idxmax()]
worst_stage = material_df_reset.loc[material_df_reset["sustainability_score"].idxmin()]

# --------------------------------------------------------------------------------
# BLOCKCHAIN STORAGE - Create Blockchain Record
# --------------------------------------------------------------------------------
block_data = {
    "timestamp": datetime.now().isoformat(),
    "material_info": {
        "UUID": material_uuid,
        "material_id": "MAT-" + material_uuid[:8].upper()
    },
    "lifecycle_data": {
        "stages": list(material_df_reset["Modul"]),
        "gwp_values": list(material_df_reset["GWP"].round(2)),
        "ap_values": list(material_df_reset["AP"].round(2)),
        "ep_values": list(material_df_reset["EP"].round(2)),
        "adpf_values": list(material_df_reset["ADPF"].round(2)),
        "adpe_values": list(material_df_reset["ADPE"].round(2)),
        "composite_impacts": list(material_df_reset["composite_impact"].round(4)),
        "sustainability_scores": list(material_df_reset["sustainability_score"].round(2))
    },
    "ai_predictions": {
        "lstm_model_version": "v1.0",
        "predicted_total_gwp": float(predicted_score.mean()),
        "prediction_confidence_r2": float(r2),
        "prediction_mae": float(mae),
        "prediction_rmse": float(rmse)
    },
    "optimization_results": {
        "best_performing_stage": best_stage["Modul"],
        "best_stage_score": float(best_stage["sustainability_score"]),
        "worst_performing_stage": worst_stage["Modul"],
        "worst_stage_score": float(worst_stage["sustainability_score"]),
        "overall_composite_score": float(material_df_reset["sustainability_score"].mean())
    },
    "metadata": {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "LSTM-64",
        "training_samples": n_samples,
        "indicators_used": indicators,
        "normalization_method": "Min-Max"
    }
}

# Generate blockchain hash
block_json = json.dumps(block_data, sort_keys=True)
block_hash = hashlib.sha256(block_json.encode()).hexdigest()
block_data["block_hash"] = block_hash

# Save to JSON file
output_filename = f"blockchain_record_{material_uuid[:8]}.json"
with open(output_filename, 'w') as f:
    json.dump(block_data, f, indent=2)

print("\n" + "=" * 80)
print("BLOCKCHAIN RECORD CREATED")
print("=" * 80)
print(f"Block Hash: {block_hash}")
print(f"Material ID: {block_data['material_info']['material_id']}")
print(f"Overall Score: {block_data['optimization_results']['overall_composite_score']:.2f}")
print(f"File Saved: {output_filename}")
print("=" * 80)

# --------------------------------------------------------------------------------
# ACTUAL VS PREDICTED PLOT WITH REGRESSION LINE
# --------------------------------------------------------------------------------
print("\nGenerating Actual vs Predicted Visualizations...")

# Flatten arrays for plotting
y_actual_flat = y_lstm.flatten()
y_predicted_flat = predicted_score.flatten()

# Calculate regression line
slope, intercept, r_value, p_value, std_err = linregress(y_actual_flat, y_predicted_flat)
regression_line = slope * y_actual_flat + intercept

# PLOT 1: Actual vs Predicted with Regression Line
fig, ax = plt.subplots(figsize=(12, 10))

# Scatter plot
scatter = ax.scatter(y_actual_flat, y_predicted_flat,
                     alpha=0.6, s=50, c='#3498db',
                     edgecolors='black', linewidth=0.5,
                     label='Data Points')

# Perfect prediction line (y=x)
min_val = min(y_actual_flat.min(), y_predicted_flat.min())
max_val = max(y_actual_flat.max(), y_predicted_flat.max())
ax.plot([min_val, max_val], [min_val, max_val],
        'r--', linewidth=3, label='Perfect Prediction (y=x)', alpha=0.8)

# Regression line
ax.plot(y_actual_flat, regression_line,
        'g-', linewidth=3, label=f'Regression Line (y={slope:.3f}x+{intercept:.3f})',
        alpha=0.8)

# Add statistics box
stats_text = f'R² Score: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nSamples: {n_samples}'
ax.text(0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='black', linewidth=2, alpha=0.9),
        fontweight='bold')

# Add blockchain info
blockchain_text = f'Block Hash: {block_hash[:32]}...'
ax.text(0.05, 0.05, blockchain_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue',
                  edgecolor='black', linewidth=1.5, alpha=0.8),
        family='monospace',
        fontweight='bold')

ax.set_xlabel('Actual Sustainability Score', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Sustainability Score', fontsize=14, fontweight='bold')
ax.set_title('LSTM Model Performance: Actual vs Predicted Sustainability Scores\n' +
             'with Regression Analysis',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)

# Equal aspect ratio
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('actual_vs_predicted_with_regression.png', dpi=300, bbox_inches='tight')
print("✓ Saved: actual_vs_predicted_with_regression.png")

# PLOT 2: Residual Plot
fig2, ax2 = plt.subplots(figsize=(12, 7))

residuals = y_actual_flat - y_predicted_flat

ax2.scatter(y_predicted_flat, residuals,
            alpha=0.6, s=50, c='#e74c3c',
            edgecolors='black', linewidth=0.5)

# Zero line
ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Residual')

# Mean residual line
mean_residual = np.mean(residuals)
ax2.axhline(y=mean_residual, color='blue', linestyle=':', linewidth=2,
            label=f'Mean Residual ({mean_residual:.4f})')

# Confidence bands (±2 std dev)
std_residual = np.std(residuals)
ax2.axhline(y=2*std_residual, color='orange', linestyle='-.', linewidth=1.5,
            alpha=0.7, label=f'±2σ ({2*std_residual:.4f})')
ax2.axhline(y=-2*std_residual, color='orange', linestyle='-.', linewidth=1.5, alpha=0.7)

ax2.set_xlabel('Predicted Sustainability Score', fontsize=13, fontweight='bold')
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=13, fontweight='bold')
ax2.set_title('Residual Analysis - Model Error Distribution\n' +
              f'Block Hash: {block_hash[:32]}...',
              fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=1)

plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: residual_plot.png")

# PLOT 3: Prediction Error Distribution
fig3, ax3 = plt.subplots(figsize=(12, 7))

ax3.hist(residuals, bins=50, color='#9b59b6', alpha=0.7,
         edgecolor='black', linewidth=1.2)

# Add normal distribution overlay
from scipy.stats import norm
mu, sigma = norm.fit(residuals)
x_hist = np.linspace(residuals.min(), residuals.max(), 100)
y_hist = norm.pdf(x_hist, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 50

ax3.plot(x_hist, y_hist, 'r-', linewidth=3,
         label=f'Normal Fit (μ={mu:.4f}, σ={sigma:.4f})')

# Add mean line
ax3.axvline(x=mu, color='green', linestyle='--', linewidth=2,
            label=f'Mean Error ({mu:.4f})')

ax3.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax3.set_title('Prediction Error Distribution Histogram\n' +
              f'Block Hash: {block_hash[:32]}...',
              fontsize=14, fontweight='bold', pad=15)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle=':', linewidth=1, axis='y')

plt.tight_layout()
plt.savefig('error_distribution_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_distribution_histogram.png")

# --------------------------------------------------------------------------------
# BLOCKCHAIN VISUALIZATIONS - BAR AND LINE PLOTS
# --------------------------------------------------------------------------------

# PLOT 4: Blockchain Stored Environmental Indicators (Bar Plot)
plt.figure(figsize=(14, 7))
stages = block_data['lifecycle_data']['stages']
x_pos = np.arange(len(stages))
width = 0.15

indicators_data = {
    'GWP': block_data['lifecycle_data']['gwp_values'],
    'AP': block_data['lifecycle_data']['ap_values'],
    'EP': block_data['lifecycle_data']['ep_values'],
    'ADPF': block_data['lifecycle_data']['adpf_values'],
    'ADPE': block_data['lifecycle_data']['adpe_values']
}

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for i, (indicator, values) in enumerate(indicators_data.items()):
    offset = width * (i - 2)
    plt.bar(x_pos + offset, values, width, label=indicator,
            color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.8)

plt.xlabel('Lifecycle Stage', fontsize=13, fontweight='bold')
plt.ylabel('Environmental Impact Value', fontsize=13, fontweight='bold')
plt.title('Blockchain Stored Environmental Indicators by Lifecycle Stage\n' +
          f'Block Hash: {block_hash[:32]}...',
          fontsize=14, fontweight='bold', pad=15)
plt.xticks(x_pos, stages, fontsize=10)
plt.legend(loc='upper left', ncol=5, fontsize=10)
plt.grid(axis='y', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('blockchain_environmental_indicators_bar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: blockchain_environmental_indicators_bar.png")

# PLOT 5: Blockchain Stored Sustainability Scores (Line Plot)
plt.figure(figsize=(14, 7))
sustainability_scores = block_data['lifecycle_data']['sustainability_scores']

plt.plot(stages, sustainability_scores, marker='o', linewidth=3,
         markersize=10, color='#2ecc71', label='Sustainability Score',
         markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2ecc71')

# Add data point labels
for i, (stage, score) in enumerate(zip(stages, sustainability_scores)):
    plt.text(i, score + 3, f'{score:.1f}', ha='center', va='bottom',
             fontweight='bold', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Add median reference line
median_score = np.median(sustainability_scores)
plt.axhline(y=median_score, color='orange', linestyle='--', linewidth=2,
            alpha=0.7, label=f'Median Score ({median_score:.2f})')

plt.xlabel('Lifecycle Stage', fontsize=13, fontweight='bold')
plt.ylabel('Sustainability Score (0-100)', fontsize=13, fontweight='bold')
plt.title('Blockchain Stored Sustainability Score Trend Across Lifecycle\n' +
          f'Block Hash: {block_hash[:32]}...',
          fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':')
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig('blockchain_sustainability_scores_line.png', dpi=300, bbox_inches='tight')
print("✓ Saved: blockchain_sustainability_scores_line.png")

# PLOT 6: AI Model Performance Metrics from Blockchain (Bar Plot)
plt.figure(figsize=(12, 7))
ai_metrics = ['R² Score', 'MAE', 'RMSE', 'Predicted Total GWP']
ai_values = [
    block_data['ai_predictions']['prediction_confidence_r2'],
    block_data['ai_predictions']['prediction_mae'],
    block_data['ai_predictions']['prediction_rmse'],
    block_data['ai_predictions']['predicted_total_gwp']
]

colors_ai = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = plt.bar(ai_metrics, ai_values, color=colors_ai,
               edgecolor='black', linewidth=2, alpha=0.85)

# Add value labels
for bar, value in zip(bars, ai_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + max(ai_values) * 0.02,
             f'{value:.4f}', ha='center', va='bottom',
             fontweight='bold', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', linewidth=1.5))

plt.ylabel('Metric Value', fontsize=13, fontweight='bold')
plt.title('Blockchain Stored AI Model Performance Metrics\n' +
          f'Model: {block_data["metadata"]["model_architecture"]} | ' +
          f'Block Hash: {block_hash[:32]}...',
          fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', alpha=0.3, linestyle=':')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('blockchain_ai_metrics_bar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: blockchain_ai_metrics_bar.png")

# PLOT 7: Composite Impact Timeline (Line Plot with Filled Area)
plt.figure(figsize=(14, 7))
composite_impacts = block_data['lifecycle_data']['composite_impacts']

plt.fill_between(range(len(stages)), composite_impacts, alpha=0.3, color='#e74c3c')
plt.plot(stages, composite_impacts, marker='s', linewidth=3,
         markersize=10, color='#e74c3c', label='Composite Impact',
         markerfacecolor='white', markeredgewidth=2, markeredgecolor='#e74c3c')

# Add data point labels
for i, (stage, impact) in enumerate(zip(stages, composite_impacts)):
    plt.text(i, impact + 0.02, f'{impact:.3f}', ha='center', va='bottom',
             fontweight='bold', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Add threshold line
impact_median = np.median(composite_impacts)
plt.axhline(y=impact_median, color='purple', linestyle='--', linewidth=2,
            alpha=0.7, label=f'Median Impact ({impact_median:.3f})')

plt.xlabel('Lifecycle Stage', fontsize=13, fontweight='bold')
plt.ylabel('Composite Environmental Impact Index', fontsize=13, fontweight='bold')
plt.title('Blockchain Stored Composite Impact Progression\n' +
          f'(Lower is Better) | Block Hash: {block_hash[:32]}...',
          fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('blockchain_composite_impact_line.png', dpi=300, bbox_inches='tight')
print("✓ Saved: blockchain_composite_impact_line.png")

# PLOT 8: Optimization Results Comparison (Bar Plot)
plt.figure(figsize=(12, 7))
optimization_labels = ['Best Stage Score', 'Worst Stage Score', 'Overall Average Score']
optimization_values = [
    block_data['optimization_results']['best_stage_score'],
    block_data['optimization_results']['worst_stage_score'],
    block_data['optimization_results']['overall_composite_score']
]
optimization_stages = [
    block_data['optimization_results']['best_performing_stage'],
    block_data['optimization_results']['worst_performing_stage'],
    'All Stages'
]

colors_opt = ['#2ecc71', '#e74c3c', '#3498db']
bars = plt.bar(optimization_labels, optimization_values, color=colors_opt,
               edgecolor='black', linewidth=2, alpha=0.85)

# Add value labels and stage names
for bar, value, stage in zip(bars, optimization_values, optimization_stages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 3,
             f'{value:.2f}\n({stage})', ha='center', va='bottom',
             fontweight='bold', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', linewidth=1.5))

plt.ylabel('Sustainability Score', fontsize=13, fontweight='bold')
plt.title('Blockchain Stored Optimization Results Summary\n' +
          f'Material ID: {block_data["material_info"]["material_id"]} | ' +
          f'Block Hash: {block_hash[:32]}...',
          fontsize=14, fontweight='bold', pad=15)
plt.ylim(0, max(optimization_values) * 1.25)
plt.grid(axis='y', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('blockchain_optimization_results_bar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: blockchain_optimization_results_bar.png")

# PLOT 9: GWP Cumulative Blockchain Data (Line Plot)
fig9, ax1 = plt.subplots(figsize=(14, 7))
gwp_values = block_data['lifecycle_data']['gwp_values']
cumulative_gwp = np.cumsum(gwp_values)

# Bar plot for individual GWP
color_bar = '#3498db'
ax1.bar(stages, gwp_values, color=color_bar, alpha=0.7,
        edgecolor='black', linewidth=1.5, label='GWP per Stage')
ax1.set_xlabel('Lifecycle Stage', fontsize=13, fontweight='bold')
ax1.set_ylabel('GWP per Stage (kg CO₂ eq)', fontsize=13, fontweight='bold', color=color_bar)
ax1.tick_params(axis='y', labelcolor=color_bar)

# Line plot for cumulative GWP
ax2 = ax1.twinx()
color_line = '#e74c3c'
ax2.plot(stages, cumulative_gwp, marker='o', linewidth=3,
         markersize=10, color=color_line, label='Cumulative GWP',
         markerfacecolor='white', markeredgewidth=2, markeredgecolor=color_line)
ax2.set_ylabel('Cumulative GWP (kg CO₂ eq)', fontsize=13, fontweight='bold', color=color_line)
ax2.tick_params(axis='y', labelcolor=color_line)

# Add labels
for i, (gwp, cum_gwp) in enumerate(zip(gwp_values, cumulative_gwp)):
    ax2.text(i, cum_gwp + max(cumulative_gwp) * 0.02, f'{cum_gwp:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=9,
             color=color_line)

plt.title('Blockchain Stored GWP Distribution and Cumulative Impact\n' +
          f'Block Hash: {block_hash[:32]}...',
          fontsize=14, fontweight='bold', pad=15)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

ax1.grid(axis='y', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('blockchain_gwp_cumulative_line.png', dpi=300, bbox_inches='tight')
print("✓ Saved: blockchain_gwp_cumulative_line.png")

plt.show()

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 80)
print("Generated Plots:")
print("  1. actual_vs_predicted_with_regression.png")
print("  2. residual_plot.png")
print("  3. error_distribution_histogram.png")
print("  4. blockchain_environmental_indicators_bar.png")
print("  5. blockchain_sustainability_scores_line.png")
print("  6. blockchain_ai_metrics_bar.png")
print("  7. blockchain_composite_impact_line.png")
print("  8. blockchain_optimization_results_bar.png")
print("  9. blockchain_gwp_cumulative_line.png")
print("=" * 80)