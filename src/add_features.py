import pandas as pd
import numpy as np

def add_features(df):
    """
    Add smart features to help predict at-risk students
    
    ⚠️ CUSTOMIZE: Change column names to match your CSV!
    """
    
    print("🎨 Adding features...")
    original_cols = len(df.columns)
    
    # ============================================
    # FEATURE 1: Average Internal Marks
    # ============================================
    # If you have multiple internal mark columns (sem1, sem2, sem3...)
    internal_cols = [c for c in df.columns if 'internal' in c.lower()]
    
    if internal_cols:
        df['avg_internal_marks'] = df[internal_cols].mean(axis=1)
        print(f"  ✅ Created avg_internal_marks from {len(internal_cols)} columns")
    else:
        print("  ⚠️ No 'internal' columns found - skipping avg_internal_marks")
    
    # ============================================
    # FEATURE 2: Marks Trend (improving or dropping?)
    # ============================================
    # Change these column names to match YOUR data!
    if 'internal_marks_sem2' in df.columns and 'internal_marks_sem1' in df.columns:
        df['marks_trend'] = df['internal_marks_sem2'] - df['internal_marks_sem1']
        print("  ✅ Created marks_trend (positive = improving, negative = dropping)")
    else:
        print("  ⚠️ Can't create marks_trend - need sem1 and sem2 columns")
    
    # ============================================
    # FEATURE 3: Is Repeater? (has backlogs)
    # ============================================
    if 'num_backlogs' in df.columns:
        df['is_repeater'] = (df['num_backlogs'] > 0).astype(int)
        print("  ✅ Created is_repeater (1 = has backlogs, 0 = no backlogs)")
    else:
        df['is_repeater'] = 0
        print("  ⚠️ No 'num_backlogs' column - set is_repeater to 0 for all")
    
    # ============================================
    # FEATURE 4: Attendance Category
    # ============================================
    if 'attendance_percent' in df.columns:
        df['attendance_category'] = pd.cut(
            df['attendance_percent'],
            bins=[0, 65, 85, 100],
            labels=['low', 'medium', 'high']
        )
        print("  ✅ Created attendance_category (low/medium/high)")
    else:
        print("  ⚠️ No 'attendance_percent' column found")
    
    # ============================================
    # FEATURE 5: Income Group
    # ============================================
    if 'family_income' in df.columns:
        df['income_group'] = pd.cut(
            df['family_income'],
            bins=[-1, 50000, 150000, 10**9],
            labels=['low', 'medium', 'high']
        )
        print("  ✅ Created income_group (low/medium/high)")
    else:
        print("  ⚠️ No 'family_income' column found")
    
    # ============================================
    # FEATURE 6: Engagement Score (combination)
    # ============================================
    if 'attendance_percent' in df.columns and 'avg_internal_marks' in df.columns:
        # Normalize to 0-100 scale and combine
        df['engagement_score'] = (
            df['attendance_percent'] * 0.5 + 
            (df['avg_internal_marks'] / df['avg_internal_marks'].max() * 100) * 0.5
        )
        print("  ✅ Created engagement_score (combined attendance + marks)")
    else:
        print("  ⚠️ Can't create engagement_score - need attendance and marks")
    
    # ============================================
    # FEATURE 7: Age (if you have date of birth)
    # ============================================
    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        df['age'] = (pd.Timestamp.now() - df['date_of_birth']).dt.days / 365.25
        print("  ✅ Created age from date_of_birth")
    
    # ============================================
    # Fill any missing values in new features
    # ============================================
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.number, 'float64', 'int64']:
                # For numbers: use median
                df[col] = df[col].fillna(df[col].median())
            elif isinstance(df[col].dtype, pd.CategoricalDtype):
                # For categorical: add 'unknown' to categories first
                df[col] = df[col].cat.add_categories('unknown').fillna('unknown')
            else:
                # For text: use "unknown"
                df[col] = df[col].fillna("unknown")
    
    new_cols = len(df.columns) - original_cols
    print(f"\n✅ Added {new_cols} new features!")
    print(f"📊 Total columns now: {len(df.columns)}")
    
    return df


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60 + "\n")
    
    # Load clean data
    df = pd.read_csv("data/students_cleaned.csv")
    print(f"📥 Loaded {len(df)} students with {len(df.columns)} columns")
    
    # Add features
    df = add_features(df)
    
    # Save result
    df.to_csv("outputs/students_with_features.csv", index=False)
    print(f"\n💾 Saved to: outputs/students_with_features.csv")
    
    # Show sample
    print("\n👀 SAMPLE OF NEW DATA:")
    print(df.head(3))
    
    print("\n🎉 Feature engineering complete!")
    print("="*60 + "\n")