import torch
import torch.nn as nn

#### LOSS FUNCTIONS AND ERROR METRICS ####
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, smooth: float=1e-7) -> torch.Tensor:
  """
  Calculate the dice loss for a batch of inputs and targets

  Parameters:
  - inputs (torch.Tensor): The input tensor.
  - targets (torch.Tensor): The target tensor.
  - smooth (float): A small number to avoid division by zero. Default is 1e-5.

  Returns:
  - torch.Tensor: The dice loss.
  """
  intersection = torch.sum(targets * inputs, dim=(1,2,3))
  sum_of_squares_pred = torch.sum(torch.square(inputs), dim=(1,2,3))
  sum_of_squares_true = torch.sum(torch.square(targets), dim=(1,2,3))
  dice = (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
  return 1-dice

def f1_score(inputs: torch.Tensor, targets: torch.Tensor, epsilon: float=1e-7, treshold:float = 0.5) -> torch.Tensor:
    """
    Calculate the F1 score for a batch of inputs and targets

    Parameters:
    - inputs (torch.Tensor): The input tensor.
    - targets (torch.Tensor): The target tensor.
    - epsilon (float): A small number to avoid division by zero. Default is 1e-7.
    - treshold (float): The treshold to use for the binary classification. Default is 0.5.

    Returns:
    - torch.Tensor: The F1 score.

    """
    # Ensure tensors have the same shape
    assert inputs.shape == targets.shape

    binary_input = (inputs >= treshold).float()

    # Calculate true positives, false positives, and false negatives
    #Black (1) is considered the positive
    true_positives = torch.sum(targets * binary_input)
    false_positives = torch.sum((1 - targets) * binary_input)
    false_negatives = torch.sum(targets * (1-binary_input))

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1

#### OPTIMIZER ####
def adam_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
  """
  Wrapper for the Adam optimizer

  Parameters:
  - model (nn.Module): The model to optimize.
  - lr (float): The learning rate.
  - weight_decay (float): The weight decay. Default is 0.0.

  Returns:
  torch.optim.Optimizer: The Adam optimizer.
  """
  return torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
