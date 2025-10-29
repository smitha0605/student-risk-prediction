import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    accuracy_score
)
import joblib
import numpy as np

import sys

# Enable more detailed error messages
sys.tracebacklimit = None

print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70 + "\n")

# ============================================
# STEP 1: LOAD DATA
# ============================================
df = pd.read_csv("outputs/students_labeled.csv")
print(f"📥 Loaded {len(df)} students")

# Check if we have labels
if 'risk_label' not in df.columns:
    print("❌ ERROR: No 'risk_label' column found!")
    print("   Run create_labels.py first!")
    exit()

print(f"  ⚠️ At Risk: {df['risk_label'].sum()} students")
print(f"  ✅ Not At Risk: {(df['risk_label']==0).sum()} students")

# ============================================
# STEP 2: PREPARE FEATURES
# ============================================
print("\n🔧 Preparing features...")

# Remove columns we DON'T want the AI to use
drop_cols = [
    'student_id',      # ID doesn't predict risk
    'name',            # Name doesn't predict risk
    'email',           # Email doesn't predict risk
    'risk_label',      # This is what we're predicting!
    'dropped_out'      # Don't use if you have it (that's cheating!)
]

# Only drop columns that actually exist
drop_cols = [c for c in drop_cols if c in df.columns]

# Get feature columns
feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"  📊 Using {len(feature_cols)} features:")
for col in feature_cols:
    print(f"     - {col}")

X = df[feature_cols]
y = df['risk_label']

# ============================================
# STEP 3: CONVERT TEXT TO NUMBERS
# ============================================
print("\n🔢 Converting categories to numbers...")

# Convert categorical columns (like "male"/"female") into numbers
# This creates "dummy" columns: gender_male, gender_female
X = pd.get_dummies(X, drop_first=True)

print(f"  ✅ After conversion: {X.shape[1]} total features")

# ============================================
# STEP 4: SPLIT DATA (80% train, 20% test)
# ============================================
print("\n✂️ Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # Makes results reproducible
    stratify=y            # Keep same ratio of at-risk in both sets
)

print(f"  📚 Training set: {len(X_train)} students")
print(f"  🧪 Test set: {len(X_test)} students")

# ============================================
# STEP 5: TRAIN THE MODEL
# ============================================
print("\n🎯 Training Random Forest model...")
print("  (This may take 30-60 seconds...)")

model = RandomForestClassifier(
    n_estimators=200,     # Use 200 decision trees
    max_depth=10,         # Prevent overfitting
    min_samples_split=10,
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)

model.fit(X_train, y_train)
print("  ✅ Training complete!")

# ============================================
# STEP 6: EVALUATE PERFORMANCE
# ============================================
print("\n📊 Evaluating model...")

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of being at-risk

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

print(f"\n🎯 Overall Accuracy: {accuracy:.1%}")
print(f"🎯 ROC AUC Score: {roc_auc:.3f}")
print("   (1.0 = perfect, 0.5 = random guessing)\n")

# Detailed report
print("DETAILED CLASSIFICATION REPORT:")
print("-" * 70)
print(classification_report(
    y_test, y_pred,
    target_names=['Not At Risk', 'At Risk'],
    digits=3
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX:")
print("-" * 70)
print(f"  True Negatives  (✅ Correctly predicted NOT at risk): {cm[0,0]}")
print(f"  False Positives (❌ Wrongly predicted AT risk): {cm[0,1]}")
print(f"  False Negatives (❌ Missed at-risk student): {cm[1,0]} ⚠️ CRITICAL")
print(f"  True Positives  (✅ Correctly predicted AT risk): {cm[1,1]}")

# ============================================
# STEP 7: FEATURE IMPORTANCE
# ============================================
print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 70)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Importance']:.3f}  - {row['Feature']}")

# ============================================
# STEP 8: SAVE THE MODEL
# ============================================
print("\n💾 Saving model...")

model_package = {
    'model': model,
    'feature_columns': X.columns.tolist(),
    'feature_importance': feature_importance,
    'metrics': {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
}

joblib.dump(model_package, "models/student_risk_model.pkl")
print("  ✅ Saved to: models/student_risk_model.pkl")

print("\n" + "="*70)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*70 + "\n")