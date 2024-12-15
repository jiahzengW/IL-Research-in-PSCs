import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from alibi.explainers import IntegratedGradients
from interpret import show
import matplotlib.pyplot as plt

# Load dataset and split into train/test set
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Global feature importance analysis using SHAP
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_train)



# Compute mean SHAP values for each feature group
group_shap_values = {}
for group in feature_groups:
    group_indices = [X.columns.get_loc(feature) for feature in group]
    shap_values_group = shap_values[1][:, group_indices]  # SHAP values for class 1
    group_shap_values[tuple(group)] = np.abs(shap_values_group).mean(axis=1)

# Display SHAP-based feature group contribution analysis
print("SHAP Feature Group Contribution (Mean):")
for group, values in group_shap_values.items():
    print(f"{group}: Mean SHAP = {np.mean(values):.4f}")

# SHAP summary plot for overall feature importance
shap.summary_plot(shap_values[1], X_train, plot_type="bar")

# InterpretML's Explainable Boosting Machine (EBM) for interpretability
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
ebm_global = ebm.explain_global(name="EBM Interpretation")
show(ebm_global)

# Select a sample for local interpretation analysis
sample_idx = 0
sample = X_test.iloc[sample_idx].values.reshape(1, -1)

# Local sensitivity analysis using Alibi's Integrated Gradients
ig = IntegratedGradients(model.predict_proba)
explanation_ig = ig.explain(sample, baselines=None, n_steps=50)

# Compute Integrated Gradients contributions
ig_contribs = explanation_ig.attributions[0].sum(axis=0)
ig_contrib_df = pd.DataFrame({
    "Feature": X.columns,
    "IG Contribution": ig_contribs
}).sort_values(by="IG Contribution", ascending=False)

# Display local sensitivity analysis results
print("Integrated Gradients Local Sensitivity Analysis:")
print(ig_contrib_df.head())

# SHAP and IG comparison visualization
shap_contrib_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Mean Absolute Contribution": np.abs(shap_values[1]).mean(axis=0)
}).sort_values(by="SHAP Mean Absolute Contribution", ascending=False)

# Merge and compare SHAP and IG contributions
merged_df = pd.merge(shap_contrib_df, ig_contrib_df, on="Feature", how="inner")
merged_df.plot(x="Feature", kind="bar", figsize=(10, 6), title="SHAP vs IG Contributions")
plt.xlabel("Feature")
plt.ylabel("Contribution")
plt.show()

# Display combined feature group contributions and local sensitivity
print("Combined Feature Group Contribution and Local Sensitivity:")
for group, shap_mean in group_shap_values.items():
    print(f"Group {group}: Mean SHAP = {np.mean(shap_mean):.4f}")
print(ig_contrib_df.head())

