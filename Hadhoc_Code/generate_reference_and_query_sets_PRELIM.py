import pandas as pd

'''
df = pd.read_csv('Test_Data/Test_Data.csv')


seed = 420

# Sample 20 spectra per species for the reference dataset
reference_df = df.groupby('Species', group_keys=False).apply(lambda x: x.sample(20, random_state=seed))

# From the remaining data, sample 10 spectra per species for the query dataset
remaining_df = df.drop(reference_df.index)
query_df = remaining_df.groupby('Species', group_keys=False).apply(lambda x: x.sample(10, random_state=seed))

# Save the subsets to new CSV files
reference_df.to_csv('Test_Data/reference_dataset_4.csv', index=False)
query_df.to_csv('Test_Data/query_dataset_4.csv', index=False)

# Display the counts to verify
counts = pd.DataFrame({
    'Reference Set': reference_df['Species'].value_counts(),
    'Query Set': query_df['Species'].value_counts()
})

'''



# Load your full dataset
df = pd.read_csv('Test_Data/Test_Data.csv')

# List of seeds for your 5 splits
seeds = [42, 420, 2025, 1234, 9999]

for i, seed in enumerate(seeds, start=1):
    # 1) Sample 20 spectra per species for reference
    reference_df = (
        df
        .groupby('Species', group_keys=False)
        .apply(lambda x: x.sample(20, random_state=seed))
    )
    
    # 2) From the rest, sample 10 per species for query
    remaining_df = df.drop(reference_df.index)
    query_df     = (
        remaining_df
        .groupby('Species', group_keys=False)
        .apply(lambda x: x.sample(10, random_state=seed))
    )
    
    # 3) Save to disk
    ref_file = f'Test_Data/reference_dataset_split{i}.csv'
    qry_file = f'Test_Data/query_dataset_split{i}.csv'
    reference_df.to_csv(ref_file, index=False)
    query_df.to_csv(qry_file,        index=False)
    
    # 4) (Optional) print counts to verify
    print(f"Split {i} (seed={seed}):")
    print("  Reference counts:")
    print(reference_df['Species'].value_counts().sort_index().to_dict())
    print("  Query counts:")
    print(query_df    ['Species'].value_counts().sort_index().to_dict())
    print()
