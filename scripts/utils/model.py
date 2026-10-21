import torch
import torch.nn as nn

if not torch.cuda.is_available():
    import numpy as np
    from numba import njit

from utils.blossom import max_weight_matching_matrix

### MODEL ARCHITECTURES ###
class DynamicPadLayer(nn.Module):
  """
  Layer for dynamic padding
  For the RNA Unet, the input size must be divisible by a number of times equal to the stride product (stride*stride*....*stride = stride_product)
  Adds zero padding at bottom and right of the input tensor to make the input a compatible size
  """
  def __init__(self, stride_product):
    super(DynamicPadLayer, self).__init__()
    self.stride_product = stride_product

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_size = x.shape[2]
    padding = self.calculate_padding(input_size, self.stride_product)
    return nn.functional.pad(x, padding)

  def calculate_padding(self, input_size: int, stride_product: int) -> tuple:
    p = stride_product - input_size % stride_product
    return (0, p, 0, p)

class MaxPooling(nn.Module):
  """
  Wrapper for the max pooling layer
  The wrapper is needed to make the pooling layer have the same inputs as convolutional layers
  """
  def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    super(MaxPooling, self).__init__()
    self.max_pool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

  def forward(self, x):
    return self.max_pool(x)
  

class RNA_Unet(nn.Module):
    def __init__(self, channels=32, in_channels=8, output_channels=1, negative_slope = 0.01, pooling = MaxPooling):
        """
        Pytorch implementation of a Unet for RNA secondary structure prediction

        Parameters:
        - channels (int): number of channels in the first hidden layer.
        - in_channels (int): number of channels in the input layer
        - output_channels (int): number of channels in the output layer
        - negative_slope (float): negative slope for the LeakyReLU activation function
        - pooling (nn.Module): the pooling layer to use
        """
        super(RNA_Unet, self).__init__()

        self.negative_slope = negative_slope

        #Add padding layer to make input size compatible with the Unet
        self.pad = DynamicPadLayer(2**4)

        # Encoder
        self.e1 = nn.Sequential(
           nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool1 = pooling(channels, channels, kernel_size=2, stride=2)

        self.e2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*2, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*2, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool2 = pooling(channels*2, channels*2, kernel_size=2, stride=2)

        self.e3 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*4, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*4, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool3 = pooling(channels*4, channels*4, kernel_size=2, stride=2)

        self.e4 = nn.Sequential(
            nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*8, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*8, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool4 = pooling(channels*8, channels*8, kernel_size=2, stride=2)

        self.e5 = nn.Sequential(
            nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*16, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*16, channels*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*16, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        #Decoder
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels*16, channels*8, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*8, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d1 = nn.Sequential(
           nn.Conv2d(channels*16, channels*8, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*8, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*8, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels*8, channels*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*4, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d2 = nn.Sequential(
           nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*4, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*4, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*2, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d3 = nn.Sequential(
           nn.Conv2d(channels*4, channels*2, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*2, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*2, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d4 = nn.Sequential(
           nn.Conv2d(channels*2, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.out = nn.Sequential(nn.Conv2d(channels, output_channels, kernel_size=1),
                                 nn.Sigmoid())

        # Initialize weights
        self.init_weights()

    def init_weights(self):
      for layer in self.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
          gain = nn.init.calculate_gain("leaky_relu", self.negative_slope)
          nn.init.xavier_uniform_(layer.weight, gain=gain)
          nn.init.zeros_(layer.bias)
        #elif isinstance(layer, nn.BatchNorm2d):
        #  nn.init.constant_(layer.weight, 1)
        #  nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        dim = x.shape[2] #Keep track of the original dimension of the input to remove padding at the end
        x = self.pad(x)

        #Encoder
        xe1 = self.e1(x)
        x = self.pool1(xe1)

        xe2 = self.e2(x)
        x = self.pool2(xe2)

        xe3 = self.e3(x)
        x = self.pool3(xe3)

        xe4 = self.e4(x)
        x = self.pool4(xe4)

        x = self.e5(x)

        #Decoder
        x = self.upconv1(x)
        x = torch.cat([x, xe4], dim=1) #Skip connection
        x = self.d1(x)

        x = self.upconv2(x)
        x = torch.cat([x, xe3], dim=1) #Skip connection
        x = self.d2(x)

        x = self.upconv3(x)
        x = torch.cat([x, xe2], dim=1) #Skip connection
        x = self.d3(x)

        x = self.upconv4(x)
        x = torch.cat([x, xe1], dim=1) #Skip connection
        x = self.d4(x)

        x = self.out(x)

        x = x[:, :, :dim, :dim]

        return x

### POST-PROCESSING ###
def pairs(x: str, y:str) -> bool:
    if x == 'A' and y == 'U':
        return True
    if x == 'U' and y == 'A':
        return True
    if x == 'C' and y == 'G':
        return True
    if x == 'G' and y == 'C':
        return True
    if x == 'G' and y == 'U':
        return True
    if x == 'U' and y == 'G':
        return True
    if x == 'N' or y == 'N':
        return True
    return False

def prepare_postprocessing(matrix: torch.Tensor, sequence: str, device: str) -> torch.Tensor:
    """
    Takes a sequence and returns a mask tensor with 1s in the positions where a base can pair with another base and for unpaired bases.

    Parameters:
    - matrix (torch.Tensor): The matrix to prepare.
    - sequence (str): The sequence that the matrix was generated from.
    - device (str): The device to use for the matrix.

    Returns:
    - torch.Tensor: The prepared matrix.
    """
    
    N = len(sequence)
    
    m = torch.eye(N, device=device)

    #Make mask to ensure only allowed base pairs and that no sharp turns are present
    for i in range(N):
        for j in range(N):
            if abs(i-j) > 3:
                if pairs(sequence[i], sequence[j]):
                    m[i, j] = 1
    
    #Make symmetric and apply mask
    matrix = (matrix + matrix.T) / 2 * m
    
    return matrix

@njit(cache=True)
def noncrossing_dp(scores, threshold, min_unpaired):
    n = scores.shape[0]
    dp = np.zeros((n, n), dtype=np.float32)
    traceback = np.full((n, n), -1, dtype=np.int32)

    for length in range(1, n):
        for i in range(n - length):
            j = i + length

            # Leave j unpaired
            best = dp[i, j - 1]
            best_k = -1

            # Pair j with k. The two remaining intervals are independent, which guarantees that selected pairs cannot cross.
            for k in range(i, j - min_unpaired):
                weight = scores[k, j]

                # Excludes masked positions and low confidence pairs.
                if weight < threshold:
                    continue

                left = dp[i, k - 1] if k > i else 0.0
                inside = dp[k + 1, j - 1] if k + 1 <= j - 1 else 0.0
                candidate = left + inside + weight

                if candidate > best:
                    best = candidate
                    best_k = k

            dp[i, j] = best
            traceback[i, j] = best_k

    return traceback


def noncrossing_postprocessing(matrix: torch.Tensor, device: str, threshold: float = 0.5, min_unpaired: int = 0) -> torch.Tensor:

    scores = matrix.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()

    trace = noncrossing_dp(scores, threshold, min_unpaired)
    n = scores.shape[0]
    partners = np.arange(n, dtype=np.int32)

    # Use a stack to avoid recursion depth issues with large sequences
    stack = [(0, n - 1)]

    while stack: 
        i, j = stack.pop()

        if i >= j:
            continue

        k = trace[i, j]

        # If k is -1, it means that j is unpaired, so we continue to the next pair (i, j-1)
        if k < 0:
            stack.append((i, j - 1))
        else:
            partners[k] = j
            partners[j] = k

            # Add the two new pairs (i, k-1) and (k+1, j-1) to the stack to continue the traceback
            if i < k - 1:
                stack.append((i, k - 1))
            if k + 1 < j - 1:
                stack.append((k + 1, j - 1))

    output = torch.zeros_like(matrix, device=device)
    indices = torch.arange(n, device=device)
    partner_tensor = torch.from_numpy(partners).to(device=device)
    output[indices, partner_tensor] = 1

    return output

def blossom_postprocessing(matrix: torch.Tensor, device: str, treshold: float = 0.5) -> torch.Tensor: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the blossom algorithm to find the maximum weight matching in the graph representation of the matrix.
    Since the blossom algorithm does not allow for self-pairing (i.e. pairing a base with itself), the matrix is first thresholded to remove weak pairings.
    This allows for bases with no strong pairings to be left out of the pairing.
    The function has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - treshold (float): The treshold to use for the matrix.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """
    matrix = matrix.clone()
    matrix[matrix < treshold] = 0

    pairs = max_weight_matching_matrix(matrix)

    y_out = torch.zeros_like(matrix, device=device)

    for (i, j) in pairs:
        y_out[i, j] = y_out[j, i] = 1

    # Find rows where all elements are zero
    zero_rows = ~torch.any(y_out != 0, dim=1)

    # Set diagonal elements to 1 in rows where all other elements are zero
    y_out[zero_rows, torch.arange(y_out.shape[0], device=device)[zero_rows]] = 1

    return y_out

def post_process(prediction: torch.Tensor, sequence: str, postprocessing_mode: str, device: str = "cpu") -> torch.Tensor:
	assert postprocessing_mode in ["restricted", "unrestricted", "restricted_noncrossing", "unrestricted_noncrossing"], f"Invalid postprocessing mode: {postprocessing_mode}"

	restricted = postprocessing_mode.startswith("restricted")
	pseudoknots = False if "noncrossing" in postprocessing_mode else True

	if restricted:
		prediction = prepare_postprocessing(prediction, sequence, device=device)
	else:
		prediction = (prediction + prediction.T) / 2

	if pseudoknots:
		return blossom_postprocessing(prediction, device=device)
	else:
		return noncrossing_postprocessing(prediction, device=device)