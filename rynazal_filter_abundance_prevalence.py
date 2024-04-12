# code adapted from: https://github.com/ryzary/shapmat_paper/blob/v1.0.0/shapmat/abundance_filter.py
import pandas as pd

def set_abundance_to_zero(data, abundance_threshold=1e-15):
    """
    Set abundance that are < threshold to 0
    skipping the sample ID, study_name, and CRC columns
    """
    # identify columns containing relative abundance values
    cols_to_convert = data.columns[1:-1]

    # convert relative abundance values to numeric values
    data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric)

    ab_filtered_1 = data.copy()

    ab_filtered_1.loc[:, cols_to_convert][data[cols_to_convert] < abundance_threshold] = 0.0
    return ab_filtered_1


def ab_filter(data, abundance_threshold=1e-15, prevalence_threshold=0.9):
    """
    1. Set abundance that are lower than abundance_threshold to 0
    2. Remove low frequency features. Features with number of zeros > percent_threshold are removed
    Returns:
        DataFrame
    """
    ab_filtered_1 = set_abundance_to_zero(data, abundance_threshold)

    # identify columns containing relative abundance values
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

# read in file containing abundance values
df = pd.read_csv("rynazal_data.csv", header=0)

# remove study name column
df.drop(df.columns[-2], axis=1, inplace=True)

# rename first column to "Sample ID"
df.rename(columns={df.columns[0]: "Sample ID"}, inplace=True)

# apply filter functions to the dataframe
filtered_df = ab_filter(df)

# write dataframe to a csv
filtered_df.to_csv("rynazal_filtered_abundance.csv", index = False)

print("Original dataframe size:", df.shape[0], "samples,", df.shape[1], "taxa.")
print("Filtered dataframe size:", filtered_df.shape[0],"samples,", filtered_df.shape[1], "taxa.")

num_zeros = (filtered_df['CRC'] == 0).sum()

print("Number of CRC samples:",num_zeros)
print("Number of control samples:",802 - num_zeros)
