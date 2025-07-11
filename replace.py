import pandas as pd
import random

# Load the CSV file
file_name = 'chunk1.csv'  # Make sure to include the correct file extension
df = pd.read_csv(file_name)

# Check the first few rows to identify the column name for acquisition costs
print(df.head())

# Replace 'acq_cost' with the actual column name in your CSV
column_name = 'Acquisition_Cost'  # Change this to the actual column name

# Generate new acquisition costs
df[column_name] = [random.randint(10000, 17080) for _ in range(len(df))]

# Save the updated DataFrame back to CSV
df.to_csv(file_name, index=False)

print("Acquisition costs updated successfully.")
