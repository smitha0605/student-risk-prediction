import pandas as pd

# Load your data
df = pd.read_csv("data/students_cleaned.csv")

print("="*60)
print("DATA INSPECTION REPORT")
print("="*60)

# How many students?
print(f"\n📊 Total Students: {len(df)}")
print(f"📊 Total Columns: {len(df.columns)}")

# What columns do we have?
print("\n📋 COLUMN NAMES:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# Show first few rows
print("\n👀 FIRST 5 ROWS:")
print(df.head())

# Check data types
print("\n🔢 DATA TYPES:")
print(df.dtypes)

# Check for missing values
print("\n⚠️ MISSING VALUES:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅ No missing values!")
else:
    print(missing[missing > 0])

# Basic statistics
print("\n📈 NUMERIC COLUMN STATISTICS:")
print(df.describe())

print("\n" + "="*60)