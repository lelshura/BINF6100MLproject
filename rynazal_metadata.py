import pandas as pd

# Create a list of metadata file names
metadata_file_paths = [
    'VogtmannE_2016_metadata.tsv',
    'WirbelJ_2018_metadata.tsv',
    'YachidaS_2019_metadata.tsv',
    'YuJ_2015_metadata.tsv',
    'ZellerG_2014_metadata.tsv'
]

# Load the filtered bacterial abundance dataset
df_abundance = pd.read_csv('rynazal_filtered_abundance.csv', header=0)

# Initialize a list to hold dataframes containing relevant columns for each metadata file
metadata_frames = []

# Loop through each metadata file and extract relevant columns
for file_path in metadata_file_paths:
    # Read the metadata file
    metadata = pd.read_csv(file_path, sep='\t', header=0)

    # Make 'SampleID' a column instead of the index and name it
    metadata.reset_index(inplace=True)
    metadata.rename(columns={'index': 'Sample ID'}, inplace=True)

    # Select only the required columns
    relevant_columns = metadata[['Sample ID', 'age', 'gender', 'country', 'BMI']]

    # Append the metadata to the list
    metadata_frames.append(relevant_columns)

# Combine all metadata into a single dataframe
combined_metadata = pd.concat(metadata_frames, ignore_index=True)


# Merge the main dataset with the combined metadata
merged_dataset = pd.merge(df_abundance, combined_metadata, on='Sample ID', how='left')

# Move the 'CRC' column to the end of the dataframe
crc = merged_dataset.pop('CRC')
merged_dataset['CRC'] = crc

# Output the shape of the merged dataset to see how many rows and columns it contains
print(merged_dataset.shape)

# Calculate the number of rows with any NA values
na_row_count = merged_dataset.isna().any(axis=1).sum()
print(f"Number of rows with at least one NA value: {na_row_count}")

# Calculate the number of NA values in each column
na_counts = merged_dataset.isna().sum()
na_counts_df = na_counts.reset_index()
na_counts_df.columns = ['Column Name', 'NA Count']
print(na_counts_df)

# Remove rows with any NA values
cleaned_merged_dataset = merged_dataset.dropna()

# Compare the shape before and after cleaning NA values
print(df_abundance.shape)
print(cleaned_merged_dataset.shape)

cleaned_merged_dataset.to_csv("rynazal_abundance_metadata.csv", index = False)