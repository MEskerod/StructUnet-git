import pickle, torch

from collections import namedtuple, defaultdict
from Bio import SeqIO

from torch.utils.data import Dataset

import numpy as np
import random as rd

RNA = namedtuple('RNA', 'input output length family name sequence')

### HANDLING INPUT FILES ###
def read_ct(file: str) -> tuple:
    """
    Takes a .ct file and returns the sequence as a string and a list pairing state of each base (0 = unpaired)

    Parameters:
    - file (str): The path to the .ct file.

    Returns:
    - tuple: A tuple containing the sequence as a string and a list of integers representing the pairing state of each base.
    """
    sequence = ""
    pairs = []

    with open(file, 'r') as f:
        lines = [line.split() for line in f.readlines()]

    #Remove header - if any
    header_lines = 0
    for line in lines:
        if line[0] == '1':
                break
        else:
            header_lines += 1

    lines = lines[header_lines:]

    for line in lines:
        sequence += line[1].upper() #Make sure the sequence is in upper case
        pairs.append(int(line[4])-1) if line[4] != '0' else pairs.append(None)

    # Make sure the sequence is the same length as the pairs list
    assert len(sequence) == len(pairs)

    return sequence, pairs

def prepare_sequence(sequence: str) -> str: 
    """
    Function that prepares a sequence for input to the model.
    The function converts the sequence to uppercase and replaces all Ts with Us.
    The function also checks that the sequence only contains valid characters.

    Parameters:
    - sequence (str): The sequence to prepare.

    Returns:
    - str: The prepared sequence.
    """
    valid_characters = ['A', 'U', 'C', 'G', 'N', 'M', 'Y', 'W', 'V', 'K', 'R', 'I', 'X', 'S', 'D', 'P', 'B', 'H']
    sequence = sequence.upper().replace('T', 'U')

    for char in sequence:
        if char not in valid_characters:
            raise ValueError(f"Invalid character '{char}' found in the sequence.")

    return sequence

def get_indices(ratios: tuple) -> tuple:
  """
  Returns indices for splitting 10 elements into three groups based on the ratios provided

  Parameters:
  - ratios (tuple): A tuple of three floats representing the ratios for the three groups.

  Returns:
  - tuple: A tuple of three lists of indices.
  """ 
  rd.seed(42)
  
  numbers = list(range(10))
  rd.shuffle(numbers)

  count1 = int(10 * ratios[0])
  count2 = int(10 * ratios[1])

  group1 = numbers[:count1]
  group2 = numbers[count1: count1+count2]
  group3 = numbers[count1+count2:]

  return group1, group2, group3

def file_length(file: str) -> int:
  """
  Returns the length of a sequence based on the value stored in the pickle file

  Parameters:
  - file (str): The path to the pickle file.

  Returns:
  - int: The length of the sequence.
  """
  return int(pickle.load(open(file, 'rb')).length)

def split_data(file_list: list, train_ratio: float = 0.8, validation_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
  """
  Split a list of files into three groups based on the provided ratios
  Returns three lists of files

  Parameters:
  - file_list (list): A list of file paths.
  - train_ratio (float, optional): The ratio for the training set. Default is 0.8.
  - validation_ratio (float, optional): The ratio for the validation set. Default is 0.1.
  - test_ratio (float, optional): The ratio for the test set. Default is 0.1.

  Returns:
  - tuple: A tuple of three lists of file paths in the order (train, valid, test).
  """
  #Test that the ratios sum to 1
  assert int((train_ratio + validation_ratio + test_ratio)) == 1
  
  #Get indices for splitting the data
  train_indices, valid_indices, test_indices = get_indices([train_ratio, validation_ratio, test_ratio])
  
  train, valid, test = [], [], []

  #Make a dictionary with the family as key and a list of files as value
  family_data = defaultdict(list)
  for file in file_list:
    family = pickle.load(open(file, 'rb')).family
    family_data[family].append((file))

  #Sort files within each family based on length
  for family, files in family_data.items():
    N = len(files)
    files.sort(key=file_length)

    #Used the optained indices to split batches of 10 files into train, validation and test
    for i in range(0, N, 10): 
      train_idx = [n+i for n in train_indices]
      valid_idx = [n+i for n in valid_indices]
      test_idx = [n+i for n in test_indices]

      train.extend([files[i] for i in train_idx if i < N])
      valid.extend(files[i] for i in valid_idx if i < N)
      test.extend(files[i] for i in test_idx if i < N)

  return train, valid, test

def read_fasta(input: str) -> list:
    """
    Reads in a FASTA-file and returns a list of sequences and their IDs

    Parameters: 
    - input: path to input sequence

    Returns:
    - list: A list of tuples containing the sequence and the ID.
    """
    records = list(SeqIO.parse(input, 'fasta'))

    return [(prepare_sequence(str(record.seq)), record.id) for record in records]

### PREPARING DATA ###
class SequenceOneHotEncoder:
    def __init__(self) -> None:
        self.seq_dict = {
            'A':np.array([1,0,0,0], dtype=np.float32),
            'U':np.array([0,1,0,0], dtype=np.float32),
            'C':np.array([0,0,1,0], dtype=np.float32),
            'G':np.array([0,0,0,1], dtype=np.float32),
            'N':np.array([0,0,0,0], dtype=np.float32),
            'M':np.array([1,0,1,0], dtype=np.float32),
            'Y':np.array([0,1,1,0], dtype=np.float32),
            'W':np.array([1,0,0,0], dtype=np.float32),
            'V':np.array([1,0,1,1], dtype=np.float32),
            'K':np.array([0,1,0,1], dtype=np.float32),
            'R':np.array([1,0,0,1], dtype=np.float32),
            'I':np.array([0,0,0,0], dtype=np.float32),
            'X':np.array([0,0,0,0], dtype=np.float32),
            'S':np.array([0,0,1,1], dtype=np.float32),
            'D':np.array([1,1,0,1], dtype=np.float32),
            'P':np.array([0,0,0,0], dtype=np.float32),
            'B':np.array([0,1,1,1], dtype=np.float32),
            'H':np.array([1,1,1,0], dtype=np.float32)}
    
    def encode(self, sequence: str) -> np.ndarray:
        """
        Convert a sequence to a onehot representation

        Parameters:
        - sequence (str): The sequence to convert.

        Returns:
        - np.array: A 3D numpy array with shape (4, len(sequence), 1).
        
        """
        onehot = np.array([self.seq_dict[base] for base in sequence])
        return onehot

def make_matrix_from_sequence_8(sequence: str) -> torch.Tensor:
    """
    A sequence is converted to a matrix containing all the possible base pairs
    Each pair in encoded as a onehot vector.
    Unpaired are the bases on the diagonal, representing the unpaired/unfolded sequence

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - torch.Tensor: A 3D tensor with shape (8, len(sequence), len(sequence)).
    """
    coding = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],  # invalid pairing
        [0, 1, 0, 0, 0, 0, 0, 0],  # unpaired
        [0, 0, 1, 0, 0, 0, 0, 0],  # GC
        [0, 0, 0, 1, 0, 0, 0, 0],  # CG
        [0, 0, 0, 0, 1, 0, 0, 0],  # UG
        [0, 0, 0, 0, 0, 1, 0, 0],  # GU
        [0, 0, 0, 0, 0, 0, 1, 0],  # UA
        [0, 0, 0, 0, 0, 0, 0, 1],  # AU
    ], dtype=np.float32)

    basepairs = ["GC", "CG", "UG", "GU", "UA", "AU"]

    N = len(sequence)

    # Create an array filled with "invalid pairing" vectors
    matrix = np.tile(coding[0], (N, N, 1))

    # Update the diagonal with "unpaired" vectors
    matrix[np.arange(N), np.arange(N), :] = coding[1]
    
    # Update base pair positions directly
    for i, j in np.ndindex(N, N):
        pair = sequence[i] + sequence[j]
        if pair in basepairs and abs(i-j) >=4:
            matrix[i, j, :] = coding[basepairs.index(pair)+2]

    return torch.from_numpy(matrix.transpose(2, 0, 1))

BaseEncoder = SequenceOneHotEncoder()

def make_matrix_from_sequence_16(sequence: str) -> torch.Tensor:
    """
    A sequence is converted to a matrix containing all the possible base pairs resulting in a 3D matrix with 16 layers
    Each pair in encoded as a onehot vector.

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - torch.Tensor: A 3D tensor with shape (16, len(sequence), len(sequence)).
    """
    x = BaseEncoder.encode(sequence)
    X =  np.kron(x, x).reshape((len(sequence), len(sequence), 16))

    return torch.from_numpy(X).permute(2, 0, 1)

def make_matrix_from_basepairs(pairs: list, unpaired: bool = True) -> torch.Tensor:
    """
    Takes a list of all which base each position in the sequence is paired with. If a base is unpaired pairs[i] = 0.
    From the list a 2D matrix is made, with each cell coresponding to a base pair encoded as 1 and unpaired bases encoded as 1 at the diagonal

    Parameters:
    - pairs (list): A list of integers representing the pairing state of each base.
    - unpaired (bool, optional): If True the unpaired bases are encoded as 1 at the diagonal. Default is True.

    Returns:
    - torch.Tensor: A 2D tensor with shape (len(pairs), len(pairs)).
    """

    N = len(pairs)
    matrix = torch.zeros((N,N), dtype=torch.float32)

    for i, j in enumerate(pairs):
        if isinstance(j, int):
            matrix[i, j] = 1
        elif unpaired:
            matrix[i, i] = 1

    return matrix


### HANDLING DATA IN PYTORCH ###
class ImageToImageDataset(Dataset):
    """
    Dataset class for image to image translation
    For each sample, the dataset returns a tuple of input and output images
    Is initialized with a list of file paths to pickle files containing the data
    """
    def __init__(self, file_list: list) -> None:
        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
      data = pickle.load(open(self.file_list[idx], 'rb'))
      
      input_image = data.input
      output_image = data.output 

      return input_image, output_image

### HANDLING OUTPUT ###
def write_ct(outputfile: str, sequence: str, output: torch.Tensor, seq_name: str) -> None:
   """
   Writes the output to a ct file.

   Parameters:
    - outputfile (str): The output file to write to.
    - sequence (str): The sequence that the output was generated from.
    - output (torch.Tensor): The structure in matrix format.
    - seq_name (str): The name of the sequence.

    Returns:
    - None
   """
   pairs = torch.nonzero(output)

   ct = [[str(i+1), sequence[i], str(i), str(i+2), str(0), str(i+1)] for i in range(len(sequence))]
   
   for row, col in pairs:
       if not row == col:
        ct[row][4] = str(col.item()+1)

   with open(outputfile, 'w') as f:
       f.write(f'{len(sequence)}\tENERGY =\t?\t{seq_name}\n')
       f.write('\n'.join(['\t'.join(line) for line in ct]))
       f.write('\n')



def write_bpseq(outputfile: str, sequence: str, output: torch.Tensor, seq_name: str) -> None:
    """
    Writes the output to a bpseq file.

    Parameters:
    - outputfile (str): The output file to write to.
    - sequence (str): The sequence that the output was generated from.
    - output (torch.Tensor): The structure in matrix format.
    - seq_name (str): The name of the sequence (not used).

    Returns:
    - None
    """
    pairs = torch.nonzero(output)

    bpseq = [[str(i+1), sequence[i], str(0)] for i in range(len(sequence))]
    
    for row, col in pairs:
        if not row == col:
            bpseq[row][2] = str(col.item()+1)
    
    with open(outputfile, 'w') as f:
        f.write(f'Filename: {outputfile}\nOrganism: Unknown\nAccession Number: 000000\nCitation and related information available at?\n')
        f.write('\n'.join([' '.join(line) for line in bpseq]))
        f.write('\n')

def write_to_stdout(outputfile: str, sequence: str, output: torch.Tensor, seq_name: str) -> None:
    """
    Writes the output to stdout.

    Parameters:
    - outputfile (str): The output file to write to (not used).
    - sequence (str): The sequence that the output was generated from.
    - output (torch.Tensor): The structure in matrix format.
    - seq_name (str): The name of the sequence.

    Returns:
    - None
    """
    pairs = torch.nonzero(output)

    bpseq = [[str(i+1), sequence[i], str(0)] for i in range(len(sequence))]
    
    for row, col in pairs:
        if not row == col:
            bpseq[row][2] = str(col.item()+1)

    print(f"\n\n>{seq_name}\n{sequence}")
    print('\n'.join([' '.join(line) for line in bpseq]))
    print('\n')