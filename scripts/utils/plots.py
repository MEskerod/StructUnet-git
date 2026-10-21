import torch, h5py

import matplotlib.pyplot as plt
import matplotlib as mpl
import colorcet as cet
import numpy as np
import pandas as pd

from collections import defaultdict

def plot_families(file_dict: dict, output_file = None) -> None:
  """
  Plots the distribution of data (families) based on the given file dictionary.

  This function takes a dictionary `file_dict` where the keys represent data set names and the value is the path to a HDF5 file.
  It plots the distribution of data (families) based on the files in the dictionary.
  The resulting plot shows the density of each family across different data sets.

  If `output_file` is provided, the plot is saved as an image file at the specified path.

  Parameters:
  - file_dict (dict): A dictionary containing the path to a HDF5 file for each data set.
  - output_file (str, optional): The path to save the plot as an image file. Defaults to None.

  Returns:
  - None
  """
  n_files = defaultdict(int)

  # Make a dictionary of RNA families and their distribution across data sets
  RNA_families = defaultdict(lambda: defaultdict(int))
  for split, file in file_dict.items():
    with h5py.File(file, 'r') as h5_file:
      n_files[split] = len(h5_file.keys())
      for sample in h5_file.keys(): 
        family = h5_file[sample]['family'][()].decode('utf-8')  # Decode bytes to string
        RNA_families[family][split] += 1

  x = np.arange(len(RNA_families))
  width = 1/len(file_dict)*0.9
  fig, ax = plt.subplots(figsize=(10,6))
  for i, split in enumerate(file_dict.keys()):
    offset = width*i
    ax.bar(x+offset, [RNA_families[f][split]/n_files[split] for f in RNA_families], width=width, edgecolor = 'black', label = split)

  ax.grid(axis = 'y', linestyle = '--')
  ax.set_axisbelow(True)
  ax.set_xticks(x + width / 2)
  ax.set_xticklabels([" ".join(f.split('_')) for f in RNA_families], rotation = 45, ha='right')
  if len(file_dict) > 1:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.94))

  ax.set_ylabel("Density")
  ax.set_title('Distribution of data (families)')

  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight', dpi = 300)


def plot_len_histogram(file_dict: dict, output_file = None, split: bool = False) -> None:
  """
  Takes a dictionary of file paths and produces a histogram of the distribution of lengths of sequences.

  Parameters:
  - file_dict (dict): A dictionary where the keys are data set names and the values are paths to HDF5 files.
  - output_file (str, optional): The file path to save the histogram plot. If not provided, the plot will be displayed.
  - split (bool): Whether to split the histogram by data set.

  Returns:
  - None

  Example usage:
  >>> file_dict = {'Data Set 1': 'file1.h5', 'Data Set 2': 'file2.h5'}
  >>> plot_len_histogram(file_dict, output_file='histogram.png')
  """
  lengths = defaultdict(list)

  for data_split, file in file_dict.items():
    with h5py.File(file, 'r') as h5_file:
      for sample in h5_file.keys():
        length = int(h5_file[sample]['length'][()])
        lengths[data_split].append(length)

  plt.figure(figsize=(10,6))

  if split:
    data_sets, lenght_lists = zip(*lengths.items())
    data_sets = list(data_sets)
    lenght_lists = list(lenght_lists)

    plt.hist(lenght_lists, bins=50, edgecolor='black', label=data_sets, zorder = 3, density = True)
  else:
    lengths = [length for length_list in lengths.values() for length in length_list]
    plt.hist(lengths, bins=50, edgecolor='black', label = '_none', zorder = 3, density = True)

  plt.title('Distribution of data (lengths)')
  plt.xlabel('Sequence Length')
  plt.ylabel("Density")
  if split and len(file_dict) > 1:
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
  input_type = inputs.shape[1]
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


def adjacent_values(data: np.ndarray, q1: float, q3: float) -> tuple:
  """
  Calculate lower and upper adjacent values for a given data distribution.
  """
  iqr = q3 - q1
  lower = q1 - 1.5 * iqr
  upper = q3 + 1.5 * iqr
  return np.clip(lower, np.nanmin(data), q1), np.clip(upper, q3, np.nanmax(data))

def find_outliers(data: np.ndarray, lower:float, upper:float) -> list:
  """
  Return values outside the given lower/upper bounds.
  Ignores NaNs automatically.
  """
  return data[(data < lower) | (data > upper)]

def set_axis_style(ax, labels: list, ylabel: str) -> None:
  """
  Set the style of the axis in a matplotlib plot.

  Parameters:
  - ax (matplotlib.axes.Axes): The axes object to modify.
  - labels (list): The labels for the y-axis ticks.
  - ylabel (str): The label for the y-axis.

  Returns:
  None
  """
  ax.set_yticks(np.arange(1, len(labels) + 1), labels=labels, fontsize = 12)
  ax.tick_params(axis='x', labelsize = 12)
  ax.set_ylim(0.25, len(labels) + 0.75)
  ax.set_ylabel(ylabel, fontsize = 10)
  ax.set_xlabel('F1 score', fontsize = 10)
  ax.tick_params(axis= 'both', which = 'both', bottom = False, left = False)
  ax.grid(axis = 'x', linestyle = '--', alpha = 0.3, zorder = 0)
  ax.set_axisbelow(True)

def compute_violin_stats(data: np.ndarray) -> dict: 
  data = data[~np.isnan(data)]
  q1, med, q3 = np.nanpercentile(data, [25, 50, 75])
  lower, upper = adjacent_values(data, q1, q3)
  out = find_outliers(data, lower, upper)

  return {'q1': q1, 'median': med, 'q3': q3, 'lower': lower, 'upper': upper, 'outliers': out}

def violin_plot(df: pd.DataFrame, ylabel: str, cmap='cet_glasbey_dark', outputfile=None, groups=None) -> None:
  """
  Generate a violin plot for the given DataFrame.

  Parameters:
  - df (pd.DataFrame): The DataFrame containing the data to be plotted.
  - ylabel (str): The label for the y-axis.
  - cmap (str, optional): The colormap to be used for coloring the violins. Defaults to 'Accent'.
  - outputfile (str, optional): The file path to save the plot. If not provided, the plot will be displayed.
  - groups (list, optional): The groups to be plotted. If not provided, all columns will be plotted.

  Returns:
  None
  """
  #Reverse order of columns such that the first column is the top of the plot
  df = df[df.columns[::-1]]
  
  cmap = mpl.colormaps[cmap].colors

  # Get number of categories to define the size of the plot
  categories = len(df.columns)
  fig, ax = plt.subplots(figsize = (10, categories))

  clean_data = [df[col].dropna().to_numpy() for col in df.columns]
  parts = ax.violinplot(clean_data, showmeans=False, showmedians = False, showextrema = False, vert = False)

  #Change apperance of body
  for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(cmap[i])
    pc.set_edgecolor('black')
    pc.set_alpha(1)
    pc.set_zorder(2)

  stats = [compute_violin_stats(data) for data in clean_data]

  quartile1 = [s['q1'] for s in stats]
  medians = [s['median'] for s in stats]
  quartile3 = [s['q3'] for s in stats]
  whiskers_min = [s['lower'] for s in stats]
  whiskers_max = [s['upper'] for s in stats]
  outliers = [s['outliers'] for s in stats]

  #Add median
  inds = np.arange(1, len(medians) + 1)
  ax.scatter(medians, inds, marker='|', color='white', s=30, zorder=3)

  #Add value as text for mean
  for i, data in enumerate(clean_data):
    ax.text(1.005, inds[i], f"{np.nanmean(data):.3f}", ha='left', fontsize = 12)

  #Add boxes
  ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5, zorder = 2.5)
  #Add whiskers
  ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1, zorder = 2.5)

  #Add outliers
  for i, out in enumerate(outliers):
    ax.scatter(out, np.ones_like(out)* inds[i], marker = 'o', s=8, color = 'black', zorder = 2.5)

  #Set style for the axes
  labels = [' '.join(name.split('_')[:-1]) for name in df.columns]
  set_axis_style(ax, labels, ylabel) 
  if groups is not None: 
    for i in range(1, categories): 
      if groups[df.columns[i - 1]] != groups[df.columns[i]]:
        ax.axhline(i + 0.5, color = 'black', linestyle = '--', alpha = 0.5, zorder = 1, linewidth = 0.8)
  plt.box(False)

  plt.subplots_adjust(bottom=0.15, wspace=0.05)

  plt.tight_layout()

  if outputfile: 
    plt.savefig(outputfile, dpi = 300, bbox_inches = 'tight')

  plt.close()