import pandas as pd
import joblib
import numpy as np
import sys
import traceback
from sklearn.inspection import permutation_importance

# Enable detailed error reporting
sys.tracebacklimit = None

print("\n" + "="*70)
print("COMPUTING EXPLANATIONS (SHAP)")
print("="*70 + "\n")

# ============================================
# STEP 1: LOAD MODEL
# ============================================
print("📦 Loading trained model...")
model_pack = joblib.load("models/student_risk_model.pkl")
model = model_pack['model']
feature_columns = model_pack['feature_columns']
print("  ✅ Model loaded")

# ============================================
# STEP 2: LOAD DATA
# ============================================
print("\n📥 Loading data...")
df = pd.read_csv("outputs/students_labeled.csv")
print(f"  ✅ Loaded {len(df)} students")

# ============================================
# STEP 3: PREPARE DATA (exact same as training)
# ============================================
print("\n🔧 Preparing data...")

# Remove non-feature columns
drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out']
drop_cols = [c for c in drop_cols if c in df.columns]

X = df[[c for c in df.columns if c not in drop_cols]]

# Convert to numbers
X = pd.get_dummies(X, drop_first=True)

# Make sure columns match training exactly
X = X.reindex(columns=feature_columns, fill_value=0)

print(f"  ✅ Prepared {len(X)} rows with {X.shape[1]} features")

# ============================================
# STEP 4: COMPUTE FEATURE IMPORTANCE
# ============================================
print("\n🧠 Computing feature importance...")
print("  (This may take 1-3 minutes...)")

# Get model predictions
y_pred = model.predict(X)

# Compute permutation importance
r = permutation_importance(
    model, X, y_pred,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Create importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': r.importances_mean,
    'Std': r.importances_std
}).sort_values('Importance', ascending=False)

print(f"  ✅ Computed importance for {len(feature_importance)} features")

# ============================================
# STEP 5: SAVE RESULTS
# ============================================
print("\n💾 Saving explanations...")

importance_package = {
    'feature_importance': feature_importance,
    'feature_names': X.columns.tolist(),
    'importances_mean': r.importances_mean,
    'importances_std': r.importances_std
}

joblib.dump(importance_package, "outputs/feature_importance.pkl")
print("  ✅ Saved to: outputs/feature_importance.pkl")

# ============================================
# STEP 6: SHOW TOP FEATURES
# ============================================
print("\n📊 TOP FEATURE IMPORTANCE:")
print("-" * 70)

# Get top contributing features
feature_impacts = feature_importance

print("\nTop 10 most important features:")
for idx, row in feature_impacts.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")

print("\n" + "="*70)
print("🎉 FEATURE IMPORTANCE COMPUTATION COMPLETE!")
print("="*70 + "\n")