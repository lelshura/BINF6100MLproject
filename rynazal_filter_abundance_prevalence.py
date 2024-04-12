# Code adapted from: https://github.com/ryzary/shapmat_paper/blob/v1.0.0/shapmat/abundance_filter.py
import pandas as pd

def set_abundance_to_zero(data, abundance_threshold=1e-15):
    """
    Set abundances that are < threshold to 0
    skipping the sample ID, study_name, and CRC columns
    """
    # Identify columns containing relative abundance values
    cols_to_convert = data.columns[1:-1]

    # Convert relative abundance values to numeric values
    data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric)

    ab_filtered_1 = data.copy()

    ab_filtered_1.loc[:, cols_to_convert][data[cols_to_convert] < abundance_threshold] = 0.0
    return ab_filtered_1


def ab_filter(data, abundance_threshold=1e-15, prevalence_threshold=0.9):
    """
    1. Set abundances that are lower than abundance_threshold to 0
    2. Remove low frequency features. Features with number of zeros > pervalence_threshold are removed
    """
    ab_filtered_1 = set_abundance_to_zero(data, abundance_threshold)

    # Identify columns containing relative abundance values
    cols_to_filter = data.columns[1:-1]

    selected_species = []
    for species in cols_to_filter:
        species_ab = list(ab_filtered_1[species])
        n_zero = species_ab.count(0)
        percent_zeros = n_zero / len(species_ab)

        if percent_zeros < prevalence_threshold:
            selected_species.append(species)

    # Include the non-filtered columns back
    selected_columns = [data.columns[0]] + selected_species + data.columns[-1:].tolist()
    filtered_data = ab_filtered_1[selected_columns]

    return filtered_data

# Read in file containing abundance values
df = pd.read_csv("rynazal_data.csv", header=0)

# Remove study name column
df.drop(df.columns[-2], axis=1, inplace=True)

# Rename first column to "Sample ID"
df.rename(columns={df.columns[0]: "Sample ID"}, inplace=True)

# Check number of control and CRC patients
num_zeros = (df['CRC'] == 0).sum()

print("Number of CRC samples:",num_zeros)
print("Number of control samples:",len(df) - num_zeros)

'''
Use default values of function to filter dataset
'''
# Apply filter functions to the dataframe
filtered_df = ab_filter(df)

# Write dataframe to a csv file
filtered_df.to_csv("rynazal_filtered_abundance.csv", index = False)

print("Original dataframe size:", df.shape[0], "samples,", df.shape[1], "taxa.")
print("Filtered dataframe size:", filtered_df.shape[0],"samples,", filtered_df.shape[1], "taxa.")

'''
Use values from rynazal tutorial to filter dataset
'''

# Apply filter function with adjusted thresholds to dataframe
filtered_df_new = ab_filter(df, abundance_threshold=1e-7, prevalence_threshold=0.95)

# Write dataframe to a csv file
filtered_df_new.to_csv("rynazal_filtered_abundance_NEW.csv", index = False)
print("New filtered dataframe size:", filtered_df_new.shape[0],"samples,", filtered_df_new.shape[1], "taxa.")
