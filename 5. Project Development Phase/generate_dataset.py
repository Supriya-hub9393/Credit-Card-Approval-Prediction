import pandas as pd
import numpy as np

# Seed for reproducible sample pipelines
np.random.seed(42)
num_samples = 1000

# Generating standard demographic and economic arrays matching your EDA project requirements
data = {
    'gender': np.random.choice([0, 1], size=num_samples),
    'income_type': np.random.choice([0, 1, 2, 3], size=num_samples),
    'annual_income': np.random.normal(loc=55000, scale=20000, size=num_samples).astype(int),
    'employment_duration': np.random.uniform(low=0, high=15, size=num_samples).round(1),
    'education_level': np.random.choice([0, 1, 2, 3], size=num_samples),
    'past_due_status': np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02])
}

df = pd.DataFrame(data)

# Underwriting rules simulation logic to structure realistic binary label indicators
# 0 = Approved, 1 = Rejected (Due to high delinquency windows or shallow financials)
df['target'] = np.where(
    (df['past_due_status'] >= 2) | 
    ((df['annual_income'] < 28000) & (df['employment_duration'] < 1.5)), 
    1, 0
)

# Enforce clean non-negative thresholds
df['annual_income'] = df['annual_income'].clip(lower=10000)

# Export local csv output artifact
df.to_csv('credit_card_data.csv', index=False)
print("✅ Automated underwriting dataset 'credit_card_data.csv' generated inside Phase 5 folder.")
