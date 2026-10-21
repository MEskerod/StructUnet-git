import pandas as pd
import numpy as np
import utils.data as utils

rnastralign_df = pd.read_csv("data/rnastralign.csv")
archiveii_df = pd.read_csv("data/archiveii.csv")
bprna_df = pd.read_csv("data/bprna_spot.csv")

rnastralign_df['pairs'] = rnastralign_df['secondary_structure'].apply(utils.dotbracket_to_pairs)
archiveii_df['pairs'] = archiveii_df['secondary_structure'].apply(utils.dotbracket_to_pairs)
bprna_df['pairs'] = bprna_df['secondary_structure'].apply(utils.dotbracket_to_pairs)

rnastralign_df["dataset"] = "rnastralign"
print(f"rnastralign_df shape: {rnastralign_df.shape}")
archiveii_df["dataset"] = "archiveii"
print(f"archiveii_df shape: {archiveii_df.shape}")
bprna_df["dataset"] = "bprna"
print(f"bprna_df shape: {bprna_df.shape}")

# Remove all rows from archiveii_df that share the same sequence as any row in rnastralign_df
archiveii_df = archiveii_df[~archiveii_df['sequence'].isin(rnastralign_df['sequence'])]
archiveii_df = archiveii_df.drop_duplicates(subset='sequence', keep='first')
print(f"archiveii_df shape after removing duplicates seen in RNAStralign: {archiveii_df.shape}")

# Remove rows from rnastralign_df that share the same sequence as another
rnastralign_df = rnastralign_df.drop_duplicates(subset='sequence', keep='first')
print(f"rnastralign_df shape after removing duplicate sequences: {rnastralign_df.shape}")

# Concatenate the three DataFrames
print(f"Concatenating rnastralign_df, archiveii_df, and bprna_df...")
combined_df = pd.concat([rnastralign_df, archiveii_df, bprna_df], ignore_index=True)
print(f"combined_df shape before removing duplicates: {combined_df.shape}")
combined_df = combined_df.drop_duplicates(subset='sequence', keep='first')
print(f"combined_df shape after removing duplicates: {combined_df.shape}")

utils.make_fasta_from_dataframe(rnastralign_df, filename="data/splits/rnastralign.fasta")
utils.make_fasta_from_dataframe(archiveii_df, filename="data/splits/archiveii.fasta")
utils.make_fasta_from_dataframe(bprna_df, filename="data/splits/bprna.fasta")
utils.make_fasta_from_dataframe(combined_df, filename="data/splits/combined.fasta")

clusters = utils.read_clstr_file("data/splits/combined_nr80_duplicates_removed.fasta.clstr")

for cluster_id, seq_ids in clusters.items():
    for seq_id in seq_ids:
        clusters[cluster_id] = [seq_id for seq_id in seq_ids if "domain" not in seq_id]
        
# Create a new DataFrame with only the sequences from the clusters and the clusters associated with them as a new column
for cluster_id, seq_ids in clusters.items():
    combined_df.loc[combined_df['id'].isin(seq_ids), 'cluster'] = cluster_id
combined_df.to_csv("data/all_with_clusters.tsv", index=False, sep="\t")

# Split the RNAStralign dataset into train, validation, and test sets where no cluster is 
# shared between the test set and the train/validation set. 
# The split is done per family, so that each family has its own train/validation/test split.
rnastralign_only = combined_df[combined_df['dataset'] == 'rnastralign'].copy()
np.random.seed(6)  # For reproducibility
rnastralign_families = rnastralign_only['family'].unique()
unique_clusters = rnastralign_only['cluster'].dropna().unique()

for family in rnastralign_families:
    family_clusters = rnastralign_only[rnastralign_only['family'] == family]['cluster'].dropna().unique()
    # Split into train/val/test ensuring no cluster is shared
    np.random.shuffle(family_clusters)
    n = len(family_clusters)
    train_clusters = family_clusters[:int(0.8 * n)]
    # Unfortunately, the remaining clusters will be split evenly between validation and test
    # There is no way to avoid sharing clusters between these two sets without skewing the family distribution in the validation and test sets. This is a limitation of the current dataset.
    train_and_val_clusters = family_clusters[int(0.8 * n):]

    rnastralign_only.loc[rnastralign_only['cluster'].isin(train_clusters), 'split_A'] = 'train'
    # Take 50% of the remaining sequences for validation and 50% for test
    rnastralign_family_remaining = rnastralign_only[(rnastralign_only['family'] == family) & (rnastralign_only['cluster'].isin(train_and_val_clusters))]
    rnastralign_family_remaining = rnastralign_family_remaining.sample(frac=1, random_state=6)  # Shuffle
    n_remaining = len(rnastralign_family_remaining)
    val_size = n_remaining // 2
    val_indices = rnastralign_family_remaining.index[:val_size]
    test_indices = rnastralign_family_remaining.index[val_size:]
    rnastralign_only.loc[val_indices, 'split_A'] = 'val'
    rnastralign_only.loc[test_indices, 'split_A'] = 'test'

# A version of the datasets that is just a random split of the RNAStralign dataset into train, validation, and test sets (80%, 10%, 10%) only considering families.
rnastralign_families = rnastralign_only['family'].unique()
n = len(rnastralign_families)

np.random.seed(6)  # For reproducibility
for family in rnastralign_families:
    rna_family_temp = rnastralign_only[rnastralign_only['family'] == family]
    family_indices = rna_family_temp.index.to_numpy()
    np.random.shuffle(family_indices)
    n_family = len(family_indices)
    train_indices = family_indices[:int(0.8 * n_family)]
    val_indices = family_indices[int(0.8 * n_family):int(0.9 * n_family)]
    test_indices = family_indices[int(0.9 * n_family):]

    rnastralign_only.loc[train_indices, 'split_B'] = 'train'
    rnastralign_only.loc[val_indices, 'split_B'] = 'val'
    rnastralign_only.loc[test_indices, 'split_B'] = 'test'

### Save datasets with and without splits
rnastralign_only.to_csv("data/splits/rnastralign_with_splits.tsv", index=False, sep="\t")

archiveii_only = combined_df[combined_df['dataset'] == 'archiveii'].copy()
archiveii_only.to_csv("data/benchmarks/archiveii_no_rnastralign_duplicates.tsv", index=False, sep="\t")

rnastralign_only_without_splits = combined_df[combined_df['dataset'] == 'rnastralign'].copy()
rnastralign_only_without_splits = rnastralign_only_without_splits.drop_duplicates(subset='cluster', keep='first', inplace=False)
rnastralign_only_without_splits.to_csv("data/benchmarks/rnastralign_no_duplicate_clusters.tsv", index=False, sep="\t")

bprna_only = combined_df[combined_df['dataset'] == 'bprna'].copy()
# Remove duplicates based on the 'cluster' column, keeping only the first 
# occurrence of each cluster for a more diverse benchmark dataset.
bprna_only = bprna_only.drop_duplicates(subset='cluster', keep='first', inplace=False)
bprna_only.to_csv("data/benchmarks/bprna_no_duplicate_clusters.tsv", index=False, sep="\t")

bprna_only_no_clusters_from_rnastralign = bprna_only[~bprna_only['cluster'].isin(rnastralign_only['cluster'])].copy()
len(bprna_only_no_clusters_from_rnastralign)
bprna_only_no_clusters_from_rnastralign.to_csv("data/benchmarks/bprna_no_clusters_from_rnastralign.tsv", index=False, sep="\t")