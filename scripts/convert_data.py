import os, sys, argparse, h5py, ast

from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

import pandas as pd

from utils.data import make_matrix_from_sequence_8, make_matrix_from_sequence_16, make_matrix_from_basepairs, sample_to_hdf5
from utils.plots import plot_families, plot_len_histogram


class Row: 
    def __init__(self, values, column_idx): 
        self.values = values
        self.column_idx = column_idx

    def __getitem__(self, column): 
        return self.values[self.column_idx[column]]


def _write_sample(outputdir: str, sample: str, data: dict) -> None:
    h5_path = os.path.join(outputdir, f'{sample}.h5')
    with h5py.File(h5_path, 'w') as h5_file:
        sample_to_hdf5(h5_file, data, sample)


def process_file(args) -> int:
    """
    Converts a file from the RNAStralign data set to a pickle file containing a namedtuple with the data.
    The data in the named tuple is the input and output matrices, the length of the sequence, the family and the name of the file.
    Output is saved in the output folder.

    Parameters:
    - sample (Row): A row from the DataFrame containing the sample data.
    - output_folder (str): The path to the folder to save the output to.
    - input_dim (int, optional): The number of channels in the input tensor being either 8 og 16. Default is 16.
    - RNAStralign (bool, optional): If set, the input data is assumed to be in the RNAStralign format. Default is True.

    Returns:
    int: 1 if the file was processed successfully, None if an error occurred.
    """
    sample, output_folder = args
    try: 
        file_id, sequence, pairs, family = sample['id'], sample['sequence'], sample['pairs'], sample['family']

        if len(sequence) == 0 or len(pairs) == 0:
            print(f"Skipping {sample['id']} due to empty sequence or pairs", file=sys.stderr)
            return
        if len(sequence) != len(pairs):
            print(f"Skipping {sample['id']} due to length mismatch between sequence and pairs", file=sys.stderr)
            return

        length = len(sequence)
        if input_dim == 16:
            input_matrix = make_matrix_from_sequence_16(sequence)
        elif input_dim == 8:
            input_matrix = make_matrix_from_sequence_8(sequence)
        output_matrix = make_matrix_from_basepairs(pairs)

        if input_matrix.shape[-1] == 0 or output_matrix.shape[-1] == 0:
            return
        
        sample_data = {'input': input_matrix, 'output': output_matrix, 'length': length, 'family': family, 'name': file_id, 'sequence': sequence}
        _write_sample(output_folder, file_id, sample_data)

        return 1

    except Exception as e: 
        # Skip this file if an unexpected error occurs during processing
        print(f"Skipping {sample['id']} due to unexpected error: {e}", file=sys.stderr)
        return


def process_and_save(data: pd.DataFrame, output_dir: str, cpus:int=1) -> None: 
    """
    Process a dataframe of sequences and structures and save the results to a specified output directory. 
    The processing is done in parallel using multiple CPU cores.

    Parameters:
    - data (pd.DataFrame): A DataFrame containing the data to process.
    - output_folder (str): The path to the folder to save the output to.
    - cpus (int): The number of CPU cores to use for parallel processing.

    Returns:
    - None
    """
    column_index = {col: idx for idx, col in enumerate(data.columns)}
    args_list = [(Row(sample, column_index), output_dir) for sample in data.itertuples(index=False, name=None)] 
    results = process_map(process_file, args_list, max_workers=cpus, chunksize=1, desc="Converting files", unit="structures")

    converted = sum([result for result in results if result is not None])  
    
    print(f"{converted} files converted", file=sys.stdout)


def parse_args():
    argparser = argparse.ArgumentParser(description='Convert RNA data to a format suitable for training and evaluation with StructUnet. The data is converted to seperate pickle files containing a namedtuple with the data. The data is split into training, validation and test sets if the split flag is set.')
    argparser.add_argument('input', type=str, help='path to a tsv file containing the data') 
    argparser.add_argument('output', type=str, help='name of the HDF5 file without extension.')
    argparser.add_argument('--input_dim', type=int, default=8, help='number of channels in the input tensor. Must be either 8 or 16. (Default = 8)', metavar='', choices=[8, 16])    
    argparser.add_argument('--cores', type=int, default=1, help='number of cores to use for parallel processing (Default = 1)', metavar='')
    argparser.add_argument('--split', type=str, default=None, help='Column name with the split information. If not provided, no splitting is done. (Default = None)', metavar='')
    argparser.add_argument('--plot', action='store_true', help='If set, plots of the family distribution and length distribution will be generated and saved in "figures". (Default = False)')
    return argparser.parse_args()
    
    
def main(): 
    args = parse_args()

    global input_dim
    input_dim = args.input_dim

    outputdir = os.path.dirname(args.output) 
    if outputdir:
        os.makedirs(outputdir, exist_ok=True)

    tmp_dir = os.environ["TMPDIR"]
    print(f"Loading data from {args.input}", file=sys.stdout)
    data = pd.read_csv(args.input, sep='\t', converters={"pairs": ast.literal_eval})
    data["family"] = data["family"].fillna("None")

    print(f'Total of {len(data)} structures to convert\n', file=sys.stdout)

    if args.split: 
        dataframes = {split: (f'{args.output}_{split}.h5', data[data[args.split] == split]) for split in data[args.split].unique()}
    else: 
        dataframes = {'all': (f'{args.output}.h5', data)}

    for split, (output_file, df) in dataframes.items():
        output_dir = f'{tmp_dir}/{split}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nConvert matrices and save to {output_file}", file=sys.stdout)
        print(f"\nConvert matrice in {split} and save to {output_file}", file=sys.stderr)
        process_and_save(df, output_dir, cpus=args.cores)

        output_dir = Path(output_dir)
        files = list(output_dir.glob('*.h5'))
        with h5py.File(output_file, 'w') as h5_file:
            for sample in tqdm(files, desc=f"Saving to HDF5", unit="files"):
                with h5py.File(sample, 'r') as h5_in:
                    sample_id = list(h5_in.keys())[0] # Only one top-level group per sample file
                    h5_in.copy(sample_id, h5_file, name=sample_id)  # Copy the group to the output HDF5 file

    if args.plot:
        os.makedirs('figures', exist_ok=True)
        file_dict = {split: output_file for split, (output_file, _) in dataframes.items()}
        plot_families(file_dict, os.path.join('figures', os.path.basename(args.input).replace('.tsv', f'{args.input_dim}_family_distribution.png')))
        if len(file_dict) > 1:
            plot_len_histogram(file_dict, os.path.join('figures', os.path.basename(args.input).replace('.tsv', f'{args.input_dim}{"_" + args.split if args.split else ""}_length_distribution.png')), split=True)
        plot_len_histogram(file_dict, os.path.join('figures', os.path.basename(args.input).replace('.tsv', f'{args.input_dim}{"_" + args.split if args.split else ""}_length_distribution_combined.png')), split=False)
        
    print("DONE", file=sys.stdout)


if __name__ == "__main__": 
    main()