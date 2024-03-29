# code adapted from: https://github.com/ryzary/shapmat_paper/blob/v1.0.0/shapmat/abundance_filter.py
import pandas as pd

def set_abundance_to_zero(data, abundance_threshold=1e-15):
    """
    Set abundance that are < threshold to 0
    skipping the sample ID, study_name, and CRC columns
    """
    # identify columns containing relative abundance values
    cols_to_convert = data.columns[1:-2]

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
    cols_to_filter = data.columns[1:-2]

    selected_species = []
    for species in cols_to_filter:
        species_ab = list(ab_filtered_1[species])
        n_zero = species_ab.count(0)
        percent_zeros = n_zero / len(species_ab)

        if percent_zeros < prevalence_threshold:
            selected_species.append(species)

    # bring the study name and CRC column to the beginning of the dataframe
    selected_columns = [data.columns[0]] + data.columns[-2:].tolist() + selected_species
    filtered_data = ab_filtered_1[selected_columns]

    return filtered_data

# read in file containing abundance values
df = pd.read_csv("bacteria_relative_abundance_concat.csv", header=0)

# apply filter functions to the dataframe
filtered_df = ab_filter(df)

# write dataframe to a csv
filtered_df.to_csv("rynazal_filtered_abundance.csv")
