import os, tempfile, shutil, tarfile, pickle, sys, multiprocessing, argparse

from tqdm import tqdm
from functools import partial
from fnmatch import fnmatch

from utils.data import read_ct, make_matrix_from_sequence_8, make_matrix_from_sequence_16, make_matrix_from_basepairs, split_data
from utils.plots import plot_families, plot_len_histogram

from collections import namedtuple
RNA = namedtuple('RNA', 'input output length family name sequence')

def get_folder_size(folder: str) -> float:
    """
    Calculate the size of a folder in GB

    Parameters:
    - folder (str): The path to the folder to calculate the size of.

    Returns:
    float: The size of the folder in GB.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    gb = total_size/1024/1024/1024
    return gb

def list_all_files(root: str, pattern = '*.ct') -> list:
    """
    List all files in a directory and its subdirectories that match a specific pattern.

    Parameters:
    - root (str): The path to the root directory.
    - pattern (str, optional): The pattern to match. Default is '*.ct'.

    Returns:
    - list: A list of file paths.
    """
    ct_files = []

    for path, subdirs, files  in os.walk(root):
        for file in files:
            if fnmatch(file, pattern):
                ct_files.append(os.path.join(path, file))
    return ct_files

def getFamily(file_name: str) -> str:
  '''
  Returns the family of a file in the RNAStralign data set, based on folder structure

  Parameters:
  - file_name (str): The path to the file to extract the family from.
  
  Returns:
  str: The family of the file.

  '''
  return '_'.join(file_name.split(os.sep)[5].split('_')[:-1])


def process_file(file: str, output_folder: str) -> int:
    """
    Converts a file from the RNAStralign data set to a pickle file containing a namedtuple with the data.
    The data in the named tuple is the input and output matrices, the length of the sequence, the family and the name of the file.
    Output is saved in the output folder.

    Parameters:
    - file (str): The path to the file to process.
    - output_folder (str): The path to the folder to save the output to.
    - input_dim (int, optional): The number of channels in the input tensor being either 8 og 16. Default is 16.
    - RNAStralign (bool, optional): If set, the input data is assumed to be in the RNAStralign format. Default is True.

    Returns:
    int: 1 if the file was processed successfully, None if an error occurred.
    """
    try: 
        with open(file, 'r') as f:
            length = int(f.readline().split()[0]) 
        if length == 0:
            return
        
        family = getFamily(file) if RNAStralign else 'unknown'
        sequence, pairs = read_ct(file)
        length = len(sequence)
        if input_dim == 16:
            input_matrix = make_matrix_from_sequence_16(sequence)
        elif input_dim == 8:
            input_matrix = make_matrix_from_sequence_8(sequence)
        output_matrix = make_matrix_from_basepairs(pairs)

        if input_matrix.shape[-1] == 0 or output_matrix.shape[-1] == 0:
            return
        
        sample = RNA(input = input_matrix, output = output_matrix, length = length, family = family, name = file, sequence = sequence)    
        
        pickle.dump(sample, open(os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '.pkl'), 'wb'))
        return 1

    except Exception as e: 
        # Skip this file if an unexpected error occurs during processing
        print(f"Skipping {file} due to unexpected error: {e}", file=sys.stderr)
        return


def process_and_save(file_list: list, output_folder: str) -> None: 
    """
    Process a list of files from the RNAStralign data set and save the output to a folder.
    The conversion is done in parallel using multiprocessing.

    Parameters:
    - file_list (list): A list of file paths to process.
    - output_folder (str): The path to the folder to save the output to.

    Returns:
    - None
    """
    total_files = len(file_list)

    os.makedirs(output_folder, exist_ok=True)

    partial_process_file = partial(process_file, output_folder=output_folder)
    
    with multiprocessing.Pool() as pool:
        #Map the process_file function to the list of files
        results = list(tqdm(pool.imap_unordered(partial_process_file, file_list), total=total_files, file=sys.stdout, desc="Processing files"))

    converted = sum([result for result in results if result is not None])  
    
    print(f"\n\n{converted} files converted", file=sys.stdout)

def main(): 
    argparser = argparse.ArgumentParser(description='Convert RNA data to a format suitable for training and evaluation with StructUnet. The data is converted to seperate pickle files containing a namedtuple with the data. The data is split into training, validation and test sets if the split flag is set.')
    argparser.add_argument('input', type=str, help='path to a gunzipped tar file containing the data set') 
    argparser.add_argument('output', type=str, help='path to the folder to save the converted data to')
    argparser.add_argument('--input_dim', type=int, default=8, help='number of channels in the input tensor. Must be either 8 or 16. (Default = 8)', metavar='')
    argparser.add_argument('-s', '--split', action='store_true', help='if set, the data is split into training, validation and test sets.')
    argparser.add_argument('-r', '--ratio', nargs=3, type=float, default=[0.8, 0.1, 0.1], help='ratio for splitting the data into training, validation and test sets. (Default = 0.8 0.1 0.1)', metavar='')
    argparser.add_argument('-d', '--data_distribution', action='store_true', help='if set, the data distribution is plotted')
    argparser.add_argument('--RNAStralign', action='store_true', help='if set, the input data is assumed to be in the RNAStralign format.')


    args = argparser.parse_args()

    assert args.input_dim in [8, 16], "ERROR: Invalid input_dim. Must be either 8 or 16"

    _, valid_ratio, test_ratio = args.ratio

    global RNAStralign, input_dim
    RNAStralign = args.RNAStralign
    input_dim = args.input_dim

    temp_dir = tempfile.mkdtemp()

    try: 
        with tarfile.open(args.input, 'r:gz') as tar: 
            print("Extract files", file=sys.stdout)
            tar.extractall(temp_dir)
        
        file_list = list_all_files(temp_dir)
        print(f'Total of {len(file_list)} files where extracted\n', file=sys.stdout)

        print("Convert matrices\n", file=sys.stdout)
        process_and_save(file_list, args.output)
    
    finally:
        shutil.rmtree(temp_dir)
    
    if args.split:
        print("Splitting data", file=sys.stdout)

        file_list = [os.path.join(args.output, file) for file in os.listdir(args.output)]

        train, valid, test = split_data(file_list, validation_ratio=valid_ratio, test_ratio=test_ratio)

        print("Move test files to different folder", file=sys.stdout)

        os.makedirs(os.path.join(args.output, 'test'), exist_ok=True)
        for file in test:
            shutil.move(file, os.path.join(args.output, 'test'))
        test = [os.path.join(args.output, 'test', os.path.basename(file)) for file in test]
        pickle.dump(test, open(os.path.join(args.output, 'test.pkl'), 'wb'))

        os.makedirs(os.path.join(args.output, 'train'), exist_ok=True)
        for file in train+valid:
            shutil.move(file, os.path.join(args.output, 'train'))
        train = [os.path.join(args.output, 'train', os.path.basename(file)) for file in train]
        valid = [os.path.join(args.output, 'train', os.path.basename(file)) for file in valid]
        pickle.dump(train, open(os.path.join(args.output, 'train.pkl'), 'wb'))
        pickle.dump(valid, open(os.path.join(args.output, 'valid.pkl'), 'wb'))

        print(f"FILES FOR TRAINING: {len(train)} for training, {len(valid)} for validation", file=sys.stdout)
        print(f"FILES FOR TESTING: {len(test)}", file=sys.stdout)

        if args.data_distribution:
            print("\nPlotting data distribution", file=sys.stdout)
        
            if RNAStralign:
                plot_families({"train":train, "valid":valid, "test":test}, output_file=os.path.join(args.output, 'family_distribution.png'))
            plot_len_histogram({"train":train, "valid":valid, "test":test}, output_file=os.path.join(args.output, 'length_distribution.png'))
        
        print("DONE", file=sys.stdout)
        print(f"Size of folder for training: {get_folder_size(os.path.join(args.output, 'train')):.2f} GB", file=sys.stdout) 
        print(f"Size of folder for testing: {get_folder_size(os.path.join(args.output, 'test')):.2f} GB", file=sys.stdout)
    
    else: 
        if args.data_distribution:
            print("\nPlotting data distribution in current working directory", file=sys.stdout)
        
            if RNAStralign:
                plot_families({"all":file_list}, output_file='family_distribution.png')
            plot_len_histogram({"all":file_list}, output_file='length_distribution.png')
        
        print("DONE", file=sys.stdout)
        print(f"Size of folder with converted files: {get_folder_size(args.output):.2f} GB", file=sys.stdout)



if __name__ == "__main__": 
    main()