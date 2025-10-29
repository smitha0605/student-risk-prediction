import pandas as pd

# Load data with features
df = pd.read_csv("outputs/students_with_features.csv")

print("🏷️ Creating risk labels...")
print(f"📊 Total students: {len(df)}")

# ============================================
# CHOOSE ONE METHOD:
# ============================================

# METHOD A: If you have historical data (BEST)
if 'dropped_out' in df.columns:
    df['risk_label'] = (df['dropped_out'] == 'yes').astype(int)
    print("  ✅ Using historical dropout data")

# METHOD B: Create rules
else:
    print("  ℹ️ No historical data - using rules")
    
    # Create risk label based on multiple risk factors
    risk_score = (
        (df['attendance_rate'] < 75).astype(int) * 0.3 +  # Low attendance
        (df['Backlogs'] > 4).astype(int) * 0.3 +         # Many backlogs
        (df['cgpa'] < 6).astype(int) * 0.2 +            # Low CGPA
        (df['study_hours_per_week'] < 15).astype(int) * 0.1 +  # Low study hours
        (df['assignments_submitted'] < df['assignments_submitted'].median()).astype(int) * 0.1  # Below median assignments
    )
    
    # Mark as at risk if risk score is above 0.5 (multiple risk factors present)
    df['risk_label'] = (risk_score > 0.5).astype(int)

# Show results
at_risk_count = df['risk_label'].sum()
not_risk_count = len(df) - at_risk_count

print(f"\n📊 LABEL DISTRIBUTION:")
print(f"  ⚠️ At Risk (1): {at_risk_count} students ({at_risk_count/len(df)*100:.1f}%)")
print(f"  ✅ Not At Risk (0): {not_risk_count} students ({not_risk_count/len(df)*100:.1f}%)")

# Save
df.to_csv("outputs/students_labeled.csv", index=False)
print(f"\n💾 Saved to: outputs/students_labeled.csv")

# Show examples
print("\n👀 SAMPLE LABELED DATA:")
print(df[['student_id', 'attendance_rate', 'Backlogs', 'cgpa', 'risk_label']].head(10))