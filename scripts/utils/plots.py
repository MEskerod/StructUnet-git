import torch, pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

def plot_families(file_dict: dict, output_file = None) -> None:
  """
  Plots the distribution of data (families) based on the given file dictionary.

  This function takes a dictionary `file_dict` where the keys represent data set names and the values are lists of files.
  It plots the distribution of data (families) based on the files in the dictionary.
  The resulting plot shows the density of each family across different data sets.

  If `output_file` is provided, the plot is saved as an image file at the specified path.

  Parameters:
  - file_dict (dict): A dictionary containing file lists for each data set.
  - output_file (str, optional): The path to save the plot as an image file. Defaults to None.

  Returns:
  - None
  """
  file_dict = {key:item for key, item in file_dict.items() if len(item)>0}

  file_lists = [item for _, item in file_dict.items()]
  data_set = [key for key, _ in file_dict.items()]

  # Make a dictionary of RNA families and their distribution across data sets
  RNA_families = {}
  for index, file_list in enumerate(file_lists):
    for file in file_list:
      family = pickle.load(open(file, 'rb')).family
      if family not in RNA_families:
        RNA_families[family] = [0]*len(file_lists)
      
      RNA_families[family][index] += 1

  #Sorts the families based on the total number of sequences across all data sets
  RNA_families = sorted({family:item for family, item in RNA_families.items() if item != ([0]*len(file_lists))}.items(), key=lambda x:x[1])
  values = [[f[1][i]/len(file_lists[i]) for i in range(len(file_lists))] for f in RNA_families]
  families = [" ".join(f[0].split('_')) for f in RNA_families]

  x = np.arange(len(families))
  width = 1/len(file_dict)*0.9

  fig, ax = plt.subplots(figsize=(10,6))
  for index in range(len(file_lists)):
    offset = width*index
    ax.bar(x+offset, [v[index] for v in values], width=width, edgecolor = 'black', label = data_set[index])

  ax.grid(axis = 'y', linestyle = '--')
  ax.set_axisbelow(True)
  ax.set_xticks(x + width / 2)
  ax.set_xticklabels(families, rotation = 45, ha='right')
  if len(file_dict) > 1:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.94))

  ax.set_ylabel("Density")
  ax.set_title('Distribution of data (families)')

  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight')


def plot_len_histogram(file_dict: dict, output_file = None) -> None:
  """
  Takes a dictionary of file lists and produces a histogram of the distribution of lengths of sequences.

  Parameters:
  - file_dict (dict): A dictionary where the keys are data set names and the values are lists of file paths.
  - output_file (str, optional): The file path to save the histogram plot. If not provided, the plot will be displayed.

  Returns:
  - None

  Example usage:
  >>> file_dict = {'Data Set 1': ['file1.pkl', 'file2.pkl'], 'Data Set 2': ['file3.pkl']}
  >>> plot_len_histogram(file_dict, output_file='histogram.png')
  """

  file_dict = {key:item for key, item in file_dict.items() if len(item)>0}

  file_lists = [item for _, item in file_dict.items()]
  data_set = [key for key, _ in file_dict.items()]

  # Takes a list of .pkl files and produces a histogram of the distribution of lengths of sequences
  lengths = []
  #Get the lengths of the sequences for each data set
  for file_list in file_lists:
    l = []
    for file in file_list:
      l.append(pickle.load(open(file, 'rb')).length)
    lengths.append(l)

  plt.figure(figsize=(10,6))

  plt.hist(lengths, bins=50, edgecolor='black', label = data_set, zorder = 3, density = True)

  plt.title('Distribution of data (lengths)')
  plt.xlabel('Sequence Length')
  plt.ylabel("Density")
  if len(file_dict) > 1:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.94))
  plt.grid(axis = 'y', linestyle='--', zorder=0)
  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight', dpi = 300)

  plt.show()

def plot_training_history(train_history: list, valid_history: list, title = None, outputfile = None) -> None:
    """
    Plots the training and validation history of a model.

    Parameters:
    - train_history (list): List of training history values (loss, F1 score, etc.).
    - valid_history (list): List of validation history values (loss, F1 score, etc.).
    - title (str): Title of the plot. Is also used as label for the y-axis.
    - outputfile (str): Path to save the plot as an image file.

    Returns:
    - None
    """
    assert len(train_history) == len(valid_history)
    
    x = list(range(1, len(train_history)+1))
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x, train_history, label = 'Training')
    ax.plot(x, valid_history, label = 'Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis = 'y', linestyle = '--')
    ax.set_axisbelow(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.tight_layout()
    if outputfile:
        plt.savefig(outputfile, bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.close()

def onehot_to_image_8(array: np.ndarray) -> np.ndarray:
  """
  Converts a one-hot encoded array with 8 channels to an RGB image.

  Parameters:
  - array (np.ndarray): A 3D NumPy array with shape (height, width, 8).

  Returns:
  - np.ndarray: A 3D NumPy array with shape (height, width, 3) representing an RGB image.
  """
  channel_to_color = {0: [255, 255, 255], #invalid pairing = white
                      1: [64, 64, 64], #unpaired = gray
                      2: [0, 255, 0], #GC = green
                      3: [0, 128, 0], #CG = dark green
                      4: [0, 0, 255], #UG = blue
                      5: [0, 0, 128], #GU = dark blue
                      6: [255, 0, 0], #UA = red
                      7: [128, 0, 0]} #AU = dark red

  rgb_image = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)

  for channel, color in channel_to_color.items():
    # Select the indices where the channel has non-zero values
    indices = array[:, :, channel] > 0
    # Assign the corresponding color to those indices in the RGB image
    rgb_image[indices] = color

  return rgb_image

def onehot_to_image_16(array: np.array) -> np.ndarray:
  """
  Converts a one-hot encoded array with 16 channels to an RGB image.

  Parameters:
  - array (np.ndarray): A 3D NumPy array with shape (height, width, 16).

  Returns:
  - np.ndarray: A 3D NumPy array with shape (height, width, 3) representing an RGB image.
  """
  channel_to_color = {0: [255, 255, 255], #invalid pairing = white (AA)
                      1: [128, 0, 0], #AU = dark red
                      2: [255, 255, 255], #invalid pairing = white (AC)
                      3: [255, 255, 255], #invalid pairing = white (AG)
                      4: [255, 0, 0], #UA = red
                      5: [255, 255, 255], #invalid pairing = white (UU)
                      6: [255, 255, 255], #invalid pairing = white (UC)
                      7: [0, 0, 255], #UG = blue
                      8: [255, 255, 255], #invalid pairing = white (CA)
                      9: [255, 255, 255], #invalid pairing = white (CU)
                      10: [255, 255, 255], #invalid pairing = white (CC)
                      11: [0, 128, 0], #CG = dark green
                      12: [255, 255, 255], #invalid pairing = white (GA)
                      13: [0, 0, 128], #GU = dark blue
                      14: [0, 255, 0], #GC = green
                      15: [255, 255, 255]} #invalid pairing = white (GG)

  rgb_image = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)

  
  for channel, color in channel_to_color.items():
    # Select the indices where the channel has non-zero values
    indices = array[:, :, channel] == 1
    # Assign the corresponding color to those indices in the RGB image
    rgb_image[indices] = color

  indices = np.sum(array == 1, axis=2) > 1
  rgb_image[indices] = [255, 255, 255]

  return rgb_image

def show_matrices(inputs: torch.Tensor, observed: torch.Tensor, predicted: torch.Tensor, treshold=0.5, output_file = None) -> None:
  """
  Plots the input, observed, predicted, and binary predicted matrices side by side.

  Parameters:
  - inputs (torch.Tensor): The input matrix.
  - observed (torch.Tensor): The observed matrix.
  - predicted (torch.Tensor): The predicted matrix.
  - treshold (float): The treshold to use for the binary predicted matrix. Default is 0.5.
  - output_file (str): OPTIONAL. Path to save the plot as an image file. If None, the plot is displayed.

  Returns:
  - None
  """
  #Determines the number of channels in the input tensor
  input_type = inputs.shape[-1]
  if input_type == 8:
    onehot_to_image = onehot_to_image_8
  elif input_type == 16:
    onehot_to_image = onehot_to_image_16
  else:
    raise ValueError("Invalid input type. Expected 8 or 16 channels.")

  fig, axs = plt.subplots(1, 4, figsize=(6,2))
  
  axs[0].imshow(onehot_to_image(inputs.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()))
  axs[0].set_title("Input")

  axs[1].imshow(observed.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy(), cmap='binary')
  axs[1].set_title("Observed")

  axs[2].imshow(predicted.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy(), cmap='binary')
  axs[2].set_title("Predicted")

  predicted_binary = (predicted.permute(0, 2, 3, 1).squeeze().detach().cpu() >= treshold).float()
  axs[3].imshow(predicted_binary, cmap='binary')
  axs[3].set_title("Predicted binary")

  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight')
  else:
    plt.show()

  plt.close()

def plot_F1(df: pd.DataFrame, outputfile: str) -> None:
    """
    Plots the F1 score for each sequence length.

    Parameters:
    - df (pd.DataFrame): The dataframe with the evaluation results.
    - outputfile (str): The name of the output file.
    - method (str): The method to plot.

    Returns:
    - None
    """
    colors = mpl.colormaps['cet_glasbey_dark'].colors

    plt.figure(figsize=(12, 4))
    
    for i, (family, group) in enumerate(df.groupby('family')):
        plt.scatter(group['length'], group['f1'], label=family, s=10, color=colors[i], alpha=0.5)
    
    plt.xlabel('Length')
    plt.ylabel('F1 score')

    plt.legend(loc = 'lower right', frameon = False)
    plt.grid(linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(outputfile, bbox_inches='tight', dpi=300)