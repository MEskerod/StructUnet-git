import argparse, os

import pandas as pd 
import numpy as np 
from utils.data import dotbracket_to_pairs, make_fasta_from_dataframe, read_clstr_file

np.random.seed(6)

def parser_arguments(): 
	parser = argparse.ArgumentParser(description="Split the RNA datasets into train, validation, and test sets.")
	parser.add_argument("--rnastralign_csv", type=str, default="data/rnastralign.csv", help="Path to the RNAStralign CSV file.")
	parser.add_argument("--archiveii_csv", type=str, default="data/archiveii.csv", help="Path to the ArchiveII CSV file.")
	parser.add_argument("--bprna_csv", type=str, default="data/bprna_spot.csv", help="Path to the BP-RNA CSV file.")
	parser.add_argument("--cluster_file", type=str, default="data/splits/combined_nr80_duplicates_removed.fasta.clstr", help="Path to the cluster file.")
	return parser.parse_args()

def load_dataset(dataset):
	dataset_name = os.path.splitext(os.path.basename(dataset))[0]
	df = pd.read_csv(dataset)
	df['pairs'] = df['secondary_structure'].apply(dotbracket_to_pairs)
	print(f"{dataset_name} shape: {df.shape}") 
	return df

def split_rnastralign_by_family(rnastralign_df):
	# Split the RNAStralign dataset into train, validation, and test sets where no cluster is 
	# shared between the test set and the train/validation set.
	# The split is done per family, so that each family has its own train/validation/test split.
	for family in rnastralign_df['family'].unique():
		family_clusters = rnastralign_df[rnastralign_df['family'] == family]['cluster'].dropna().unique()
		# Split into train/val/test ensuring no cluster is shared
		np.random.shuffle(family_clusters)
		n = len(family_clusters)
		train_clusters = family_clusters[:int(0.8 * n)]
		# The reamining clusters are split evenly into validation and test sets, but the family distribution will be skewed to avoid sharing clusters between train/val/test sets
		train_and_val_clusters = family_clusters[int(0.8 * n):]
		rnastralign_df.loc[rnastralign_df['cluster'].isin(train_clusters), 'split_A'] = 'train'
		# Take 50% of the remaining clusters for validation and 50% for test
		rnastralign_family_remaining = rnastralign_df[(rnastralign_df['family'] == family) & (rnastralign_df['cluster'].isin(train_and_val_clusters))]
		rnastralign_family_remaining = rnastralign_family_remaining.sample(frac=1, random_state=6)  # Shuffle
		n_remaining = len(rnastralign_family_remaining)
		val_size = n_remaining // 2
		val_indices = rnastralign_family_remaining.index[:val_size]
		test_indices = rnastralign_family_remaining.index[val_size:]
		rnastralign_df.loc[val_indices, 'split_A'] = 'val'
		rnastralign_df.loc[test_indices, 'split_A'] = 'test'

	# A second version of the splits that is just a random split of the RNAStralign dataset, without considering clusters or families. (80% train, 10% val, 10% test)
	for family in rnastralign_df['family'].unique():
		rna_family_temp = rnastralign_df[rnastralign_df['family'] == family].copy()
		family_indices = rna_family_temp.index.to_numpy()
		np.random.shuffle(family_indices)
		n = len(family_indices)
		train_indices = family_indices[:int(0.8 * n)]
		val_indices = family_indices[int(0.8 * n):int(0.9 * n)]
		test_indices = family_indices[int(0.9 * n):]

		rnastralign_df.loc[train_indices, 'split_B'] = 'train'
		rnastralign_df.loc[val_indices, 'split_B'] = 'val'
		rnastralign_df.loc[test_indices, 'split_B'] = 'test'


def main(): 
	args = parser_arguments()
	os.makedirs('data', exist_ok=True)

	rnastralign_df = load_dataset(args.rnastralign_csv)
	archiveii_df = load_dataset(args.archiveii_csv)
	bprna_df = load_dataset(args.bprna_csv)

	# Remove duplicates from data  
	archiveii_df = archiveii_df[~archiveii_df['sequence'].isin(rnastralign_df['sequence'])].drop_duplicates(subset='sequence', keep='first') # Remove sequences from ArchiveII that are in RNAStralign
	print(f"\nArchiveII shape after removing duplicates seen in RNAStralign: {archiveii_df.shape}")

	rnastralign_df = rnastralign_df.drop_duplicates(subset='sequence', keep='first')
	print(f"RNAStralign shape after removing duplicate sequences: {rnastralign_df.shape}")

	# Concatenate all three data frames 
	print(f"\nConcatenating RNAStralign, ArchiveII, and bpRNA...")
	combined_df = pd.concat([rnastralign_df, archiveii_df, bprna_df], ignore_index=True).drop_duplicates(subset='sequence', keep='first')
	print(f"Combined shape after removing duplicates: {combined_df.shape}")

	make_fasta_from_dataframe(rnastralign_df, filename='data/rnastralign.fasta')
	make_fasta_from_dataframe(archiveii_df, filename='data/archiveii.fasta')
	make_fasta_from_dataframe(bprna_df, filename='data/bprna.fasta')
	make_fasta_from_dataframe(combined_df, filename='data/combined.fasta')


	print(f"\nReading clusters from {args.cluster_file}...")	
	clusters = read_clstr_file(args.cluster_file)

	# Create a new DataFrame with only the sequences from the clusters and the clusters associated with them as a new column
	for cluster_id, seq_ids in clusters.items():
		combined_df.loc[combined_df['id'].isin(seq_ids), 'cluster'] = cluster_id
	combined_df.to_csv("data/all_data_w_clusters.tsv", index=False, sep="\t")

	print(f"\nSplitting data and removing cluster duplicates...")
	# Split RNAStralign and save file
	rnastralign_only = combined_df[combined_df['dataset'] == 'rnastralign'].copy()
	split_rnastralign_by_family(rnastralign_only)
	rnastralign_only.to_csv("data/RNAStrAlign.tsv", index=False, sep="\t")

	# Save RNAStralign with no cluster duplicates
	rnastralign_no_cluster = combined_df[combined_df['dataset'] == 'rnastralign'].copy()
	rnastralign_no_cluster = rnastralign_no_cluster.drop_duplicates(subset='cluster', keep='first', inplace=False)
	rnastralign_no_cluster.to_csv("data/RNAStrAlign_no_cluster_duplicates.tsv", index=False, sep="\t")

	# Filter ArchiveII and bpRNA datasets to remove clusters
	archiveii_only = combined_df[combined_df['dataset'] == 'archiveii'].copy()
	archiveii_only.to_csv("data/ArchiveII.tsv", index=False, sep="\t")

	# Filter bpRNA dataset to remove cluster duplicates and clusters in RNAStralign
	bprna_only = combined_df[combined_df['dataset'] == 'bprna'].copy()
	bprna_only = bprna_only.drop_duplicates(subset='cluster', keep='first', inplace=False)
	bprna_only = bprna_only[~bprna_only['cluster'].isin(rnastralign_only['cluster'])]
	bprna_only.to_csv("data/bpRNA.tsv", index=False, sep="\t")

	print(f"\nDone!")


if __name__ == "__main__":
	main()