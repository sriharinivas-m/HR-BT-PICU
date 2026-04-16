"""
Generate Synthetic Data for Heart Rate and Body Temperature Relationship Study
Based on: Heart Rate and Body Temperature Relationship in Children Admitted to PICU: 
A Machine Learning Approach (IEEE TBME 2025)

This script generates synthetic data that mimics the characteristics described in the paper.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_picu_data(n_patients=4007, n_observations=4462):
    """
    Generate synthetic PICU data based on paper parameters.
    
    Parameters:
    - n_patients: Number of unique patients (default: 4007 as in paper)
    - n_observations: Number of HR-BT observations (default: 4462 as in paper)
    
    Returns:
    - DataFrame with synthetic data
    """
    
    data = []
    
    # Generate patient IDs
    patient_ids = list(range(1, n_patients + 1))
    
    # Age distribution based on paper (0-18 years = 0-216 months)
    # Age groups: newborn (0-28 days), infant (29 days-1 year), 
    # toddler (1-2 years), child (2-12 years), teenager (12-18 years)
    
    # Generate ages in months with realistic distribution
    # More patients in younger age groups (common in PICU)
    age_months = []
    for _ in range(n_patients):
        # Weighted towards younger ages
        age_choice = np.random.choice(
            ['newborn', 'infant', 'toddler', 'child', 'teenager'],
            p=[0.15, 0.25, 0.15, 0.30, 0.15]
        )
        
        if age_choice == 'newborn':
            age_months.append(np.random.uniform(0, 1))  # 0-1 month
        elif age_choice == 'infant':
            age_months.append(np.random.uniform(1, 12))  # 1-12 months
        elif age_choice == 'toddler':
            age_months.append(np.random.uniform(12, 24))  # 12-24 months
        elif age_choice == 'child':
            age_months.append(np.random.uniform(24, 144))  # 2-12 years
        else:  # teenager
            age_months.append(np.random.uniform(144, 216))  # 12-18 years
    
    # Create patient dictionary
    patients = {}
    for i, (pid, age_m) in enumerate(zip(patient_ids, age_months)):
        patients[pid] = {
            'age_months': age_m,
            'age_years': age_m / 12,
            'age_group': get_age_group(age_m)
        }
    
    # Generate observations
    # Each patient can have multiple observations (one per temperature range)
    # But we ensure approximately one observation per patient per BT group
    
    # Body temperature ranges: 33-33.9, 34-34.9, ..., 40-40.9 (8 ranges)
    bt_ranges = [(33 + i, 33.9 + i) for i in range(8)]  # 33-40.9
    
    # Generate observations
    observation_count = 0
    patient_bt_combinations = {}  # Track which BT ranges each patient has
    
    while observation_count < n_observations:
        # Select a random patient
        patient_id = np.random.choice(patient_ids)
        age_months = patients[patient_id]['age_months']
        age_years = patients[patient_id]['age_years']
        
        # Select a BT range (weighted towards normal ranges 36-38°C)
        bt_range_idx = np.random.choice(
            len(bt_ranges),
            p=[0.05, 0.05, 0.10, 0.15, 0.30, 0.20, 0.10, 0.05]  # More in normal range
        )
        bt_min, bt_max = bt_ranges[bt_range_idx]
        
        # Check if this patient already has an observation in this BT range
        if patient_id not in patient_bt_combinations:
            patient_bt_combinations[patient_id] = set()
        
        bt_range_key = f"{bt_min:.1f}-{bt_max:.1f}"
        if bt_range_key in patient_bt_combinations[patient_id]:
            continue  # Skip if patient already has observation in this range
        
        patient_bt_combinations[patient_id].add(bt_range_key)
        
        # Generate body temperature within range
        body_temperature = np.random.uniform(bt_min, bt_max)
        
        # Generate heart rate based on age and temperature
        # Relationship: HR decreases with age, increases with temperature
        # Base HR by age (from paper):
        if age_months < 1:  # Newborn
            base_hr = np.random.normal(145, 30)  # Mean 145, std 30
        elif age_months < 12:  # Infant
            base_hr = np.random.normal(140, 25)
        elif age_months < 24:  # Toddler
            base_hr = np.random.normal(120, 20)
        elif age_months < 144:  # Child (2-12 years)
            base_hr = np.random.normal(100, 20)
        else:  # Teenager
            base_hr = np.random.normal(80, 15)
        
        # Temperature effect: ~10-12 bpm per 1°C increase (from paper)
        # Normal temperature is ~37°C
        temp_effect = (body_temperature - 37.0) * np.random.uniform(9, 13)
        
        # Add some non-linearity (as paper suggests non-linear relationship)
        if body_temperature < 36:
            temp_effect *= 0.8  # Less effect at lower temps
        elif body_temperature > 38:
            temp_effect *= 1.2  # More effect at higher temps
        
        heart_rate = base_hr + temp_effect
        
        # Add noise
        heart_rate += np.random.normal(0, 8)
        
        # Ensure HR is within valid range (30-240 bpm)
        heart_rate = np.clip(heart_rate, 30, 240)
        
        # Temperature measurement site
        temp_site = np.random.choice(
            ['rectal', 'esophageal', 'axillary'],
            p=[0.5, 0.3, 0.2]  # Rectal most common
        )
        
        # Measurement type
        measurement_type = np.random.choice(
            ['continuous', 'manual'],
            p=[0.7, 0.3]  # More continuous measurements
        )
        
        # Generate timestamp (within study period: Aug 2018 - Oct 2022)
        start_date = datetime(2018, 8, 1)
        end_date = datetime(2022, 10, 31)
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = np.random.randint(0, days_between)
        timestamp = start_date + timedelta(days=random_days, 
                                          hours=np.random.randint(0, 24),
                                          minutes=np.random.randint(0, 60))
        
        # Comfort scores (always generate valid values for calm patients)
        # CAPD score (0-32, <9 = no delirium) - for invasively ventilated children
        # Assign based on patient characteristics (younger patients more likely to have CAPD)
        if age_months < 24:  # More common in younger patients
            capd_score = np.random.uniform(0, 8) if np.random.random() < 0.4 else np.random.uniform(0, 6)
        else:
            capd_score = np.random.uniform(0, 6) if np.random.random() < 0.2 else np.random.uniform(0, 4)
        
        # Comfort B score (6-30, 11-17 = normal) - for ventilated and sedated patients
        # Most patients in normal range (calm state)
        comfort_b_score = np.random.uniform(11, 17)
        
        # FLACC score (0-10, 0-3 = no/mild pain) - for noninvasive patients under 6 years
        if age_months < 72:  # Under 6 years
            flacc_score = np.random.uniform(0, 3)
        else:
            flacc_score = np.random.uniform(0, 2)  # Lower for older children
        
        # RASS score (-5 to +4, -5 to +1 = included) - for agitation/sedation assessment
        # Most patients in calm range
        rass_score = np.random.uniform(-5, 1)
        
        # VNS score (0-10, 0-3 = no/mild pain) - for patients over 6 years who can communicate
        if age_months >= 72:  # Over 6 years
            vns_score = np.random.uniform(0, 3)
        else:
            vns_score = np.random.uniform(0, 2)  # Lower for younger children
        
        # Medication status (most patients not on HR-affecting meds at observation time)
        on_medication = np.random.choice([0, 1], p=[0.85, 0.15])
        
        # Length of stay (average 7.30 days, std 29.28 days from paper)
        length_of_stay_days = max(1, np.random.lognormal(np.log(7.3), 1.0))
        
        # Create observation
        observation = {
            'patient_id': patient_id,
            'age_months': age_months,
            'age_years': age_years,
            'age_group': patients[patient_id]['age_group'],
            'body_temperature': round(body_temperature, 2),
            'body_temperature_range': f"{bt_min:.1f}-{bt_max:.1f}",
            'heart_rate': round(heart_rate, 1),
            'temperature_site': temp_site,
            'measurement_type': measurement_type,
            'timestamp': timestamp,
            'capd_score': round(capd_score, 1),
            'comfort_b_score': round(comfort_b_score, 1),
            'flacc_score': round(flacc_score, 1),
            'rass_score': round(rass_score, 1),
            'vns_score': round(vns_score, 1),
            'on_medication': on_medication,
            'length_of_stay_days': round(length_of_stay_days, 1)
        }
        
        data.append(observation)
        observation_count += 1
        
        # Progress indicator
        if observation_count % 500 == 0:
            print(f"Generated {observation_count}/{n_observations} observations...")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by patient_id and timestamp
    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    
    return df

def get_age_group(age_months):
    """Categorize age into groups as per paper."""
    if age_months < 1:  # 0-28 days
        return 'newborn'
    elif age_months < 12:  # 29 days - 1 year
        return 'infant'
    elif age_months < 24:  # 1-2 years
        return 'toddler'
    elif age_months < 144:  # 2-12 years
        return 'child'
    else:  # 12-18 years
        return 'teenager'

def print_data_summary(df):
    """Print summary statistics of the generated data."""
    print("\n" + "="*80)
    print("SYNTHETIC DATA SUMMARY")
    print("="*80)
    print(f"\nTotal Observations: {len(df)}")
    print(f"Total Unique Patients: {df['patient_id'].nunique()}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\n--- Age Distribution ---")
    print(df['age_group'].value_counts().sort_index())
    print(f"\nAge Statistics (months):")
    print(f"  Mean: {df['age_months'].mean():.2f} months ({df['age_months'].mean()/12:.2f} years)")
    print(f"  Min: {df['age_months'].min():.2f} months")
    print(f"  Max: {df['age_months'].max():.2f} months")
    
    print("\n--- Body Temperature Distribution ---")
    print(f"Mean: {df['body_temperature'].mean():.2f}°C")
    print(f"Min: {df['body_temperature'].min():.2f}°C")
    print(f"Max: {df['body_temperature'].max():.2f}°C")
    print(f"\nTemperature Range Distribution:")
    print(df['body_temperature_range'].value_counts().sort_index())
    
    print("\n--- Heart Rate Distribution ---")
    print(f"Mean: {df['heart_rate'].mean():.1f} bpm")
    print(f"Min: {df['heart_rate'].min():.1f} bpm")
    print(f"Max: {df['heart_rate'].max():.1f} bpm")
    print(f"Std: {df['heart_rate'].std():.1f} bpm")
    
    print("\n--- Temperature Measurement Site ---")
    print(df['temperature_site'].value_counts())
    
    print("\n--- Measurement Type ---")
    print(df['measurement_type'].value_counts())
    
    print("\n--- Observations per Patient ---")
    obs_per_patient = df.groupby('patient_id').size()
    print(f"Mean: {obs_per_patient.mean():.2f}")
    print(f"Min: {obs_per_patient.min()}")
    print(f"Max: {obs_per_patient.max()}")
    
    print("\n--- HR vs BT Relationship (by Age Group) ---")
    for age_group in df['age_group'].unique():
        group_data = df[df['age_group'] == age_group]
        if len(group_data) > 0:
            correlation = group_data['heart_rate'].corr(group_data['body_temperature'])
            print(f"{age_group:12s}: Correlation = {correlation:.3f}, "
                  f"Mean HR = {group_data['heart_rate'].mean():.1f} bpm, "
                  f"Mean BT = {group_data['body_temperature'].mean():.2f}°C")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Generating synthetic PICU data based on paper parameters...")
    print("Paper: Heart Rate and Body Temperature Relationship in Children Admitted to PICU")
    print("IEEE Transactions on Biomedical Engineering, 2025\n")
    
    # Generate data
    df = generate_synthetic_picu_data(n_patients=4007, n_observations=4462)
    
    # Print summary
    print_data_summary(df)
    
    # Verify no null values
    print("\n--- Null Value Check ---")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("[OK] No null values found in any column!")
    else:
        print("[WARNING] Found null values:")
        print(null_counts[null_counts > 0])
    
    # Check for empty columns
    print("\n--- Empty Column Check ---")
    empty_cols = []
    for col in df.columns:
        if df[col].isnull().all() or (df[col].dtype == 'object' and (df[col] == '').all()):
            empty_cols.append(col)
    if len(empty_cols) == 0:
        print("[OK] No empty columns found!")
    else:
        print(f"[WARNING] Found empty columns: {empty_cols}")
    
    # Save to CSV (try with different filename if permission error)
    output_file = "Dataset/HeartRate.csv"
    try:
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Data saved to: {output_file}")
    except PermissionError:
        output_file = "Dataset/HeartRate.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[WARNING] Permission denied for original file. Data saved to: {output_file}")
    
    # Also save a sample for quick inspection
    sample_file = "synthetic_picu_data_sample.csv"
    try:
        df.head(100).to_csv(sample_file, index=False)
        print(f"[OK] Sample (first 100 rows) saved to: {sample_file}")
    except PermissionError:
        sample_file = "synthetic_picu_data_sample_new.csv"
        df.head(100).to_csv(sample_file, index=False)
        print(f"[WARNING] Permission denied for original sample file. Sample saved to: {sample_file}")
    
    print("\nGeneration complete!")

