import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = 'Final.csv'
unsafe_dataset_only = False

df = pd.read_csv('Final.csv')[:3000]

# Create a temporary column 'label' to indicate the safety status
df['label'] = 'safe'
df.loc[:999, 'label'] = 'unsafe'

if unsafe_dataset_only:
    df = df[:1000]

# Split the data into training and testing sets with a 7:3 ratio, maintaining the unsafe:safe ratio
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])

# Drop the temporary 'label' column as it's no longer needed
train_data.drop('label', axis=1, inplace=True)
test_data.drop('label', axis=1, inplace=True)

# Save the new DataFrames to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)