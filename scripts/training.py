import os, pickle, logging, sys, argparse

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import namedtuple
import pandas as pd


from utils.plots import show_matrices, plot_training_history
from utils.data import ImageToImageDataset
from utils.model import RNA_Unet
from utils.training import dice_loss, f1_score, adam_optimizer

def fit_model(model: torch.nn.Module, train_dataset, validtion_dataset, model_path: str, patience: int = 5, lr: float = 0.01, weigth_decay: float = 0.0, optimizer =adam_optimizer, loss_function = dice_loss, epochs: int = 60, batch_size: int = 1) -> dict: 
    """
    Trains a model on a training dataset and validates it on a validation dataset.
    Is implemented with early stopping. 
    The best model is saved as 'RNA_Unet.pth'.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - train_dataset (torch.utils.data.Dataset): The training dataset.
    - validtion_dataset (torch.utils.data.Dataset): The validation dataset.
    - model_path (str): The path to save the best model.
    - patience (int): The number of epochs without improvement before stopping training. Default is 5.
    - lr (float): The learning rate for the optimizer. Default is 0.01.
    - weigth_decay (float): The weight decay for the optimizer. Default is 0.
    - optimizer (function): The optimizer to use. Default is utils.adam_optimizer.
    - loss_function (function): The loss function to use. Default is utils.dice_loss.
    - epochs (int): The number of epochs to train. Default is 60.
    - batch_size (int): The batch size for training. Default is 1.

    Returns:
    - dict: A dictionary with the training history. Keys are 'train_loss', 'train_F1', 'valid_loss', 'valid_F1'.
    """
    best_score = float('inf')
    early_stopping_counter = 0
    
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader(validtion_dataset, batch_size = batch_size)

    opt = optimizer(model, lr, weigth_decay)

    train_loss_history, train_F1_history, valid_loss_history, valid_F1_history = [], [], [], []
    start_epoch = 0

    #Only load previous training history if it exists
    if os.path.exists('training_log/training_history.csv'):
        logging.info('Loading previous training history...')
        df = pd.read_csv('training_log/training_history.csv')
        train_loss_history = df['train_loss'].tolist()
        train_F1_history = df['train_F1'].tolist()
        valid_loss_history = df['valid_loss'].tolist()
        valid_F1_history = df['valid_F1'].tolist()
        #Continue training from last epoch
        start_epoch = len(train_loss_history)
        best_score = min(valid_loss_history)
        early_stopping_counter = len(valid_loss_history) - valid_loss_history.index(best_score) - 1
        logging.info(f'Starting training from epoch {start_epoch+1}.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)

    logging.info(f"Training model with {len(train_dl)} training samples and {len(valid_dl)} validation samples. Device: {device}")

    for epoch in range(start_epoch, epochs): 
        logging.info(f"\nStarting epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(total = len(train_dl), desc = f'Training of epoch {epoch+1}', unit = 'batch')

        running_loss, running_F1 = 0.0, 0.0
        model.train()

        try:
          for input, target in train_dl: 
              input, target = input.to(device), target.unsqueeze(1) #Since the model expects a channel dimension target needs to be unsqueezed

              #Forward pass
              opt.zero_grad()
              output = model(input)

              output = output.cpu()
              loss = loss_function(output, target)
              loss = loss.to(device)
            
              #Backward pass
              loss.backward()
              opt.step()

              running_loss += loss.item()
              running_F1 += f1_score(output, target).item()

              progress_bar.update(1)
        
        except Exception as e:
          print(e, file=sys.stderr)
          print("Input shape:", input.shape, file=sys.stderr)
          print("Target shape:", target.shape, file=sys.stderr)
          sys.exit(1)    
        
        progress_bar.close()
        
        show_matrices(input, target, output, output_file = f'training_log/matrix_example_epoch{epoch+1}.png')
        
        #Validation loss (only after each epoch)
        logging.info("Start validation...")
        progress_bar = tqdm(total = len(valid_dl), desc = f'Validation of epoch {epoch+1}', unit = 'sequence')
        valid_loss, valid_F1 = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for input, target in valid_dl: 
                input, target = input.to(device), target.unsqueeze(1)

                output = model(input)
                output = output.cpu()
                valid_loss += loss_function(output, target).item()
                valid_F1 += f1_score(output, target).item()
                progress_bar.update(1)
        progress_bar.close()
        
        val_loss = valid_loss/len(valid_dl)
        
        train_loss_history.append(running_loss/len(train_dl))
        train_F1_history.append(running_F1/len(train_dl))
        valid_loss_history.append(val_loss)
        valid_F1_history.append(valid_F1/len(valid_dl))

        logging.info(f"Epoch {epoch+1}/{epochs}: Train loss: {train_loss_history[-1]:.4f}, Train F1: {train_F1_history[-1]:.4f}, Validation loss: {valid_loss_history[-1]:.4f}, Validation F1: {valid_F1_history[-1]:.4f}")
        if epoch > 0:
            plot_training_history(train_loss_history, valid_loss_history, title = 'Loss', outputfile = 'training_log/loss_history.png')
            plot_training_history(train_F1_history, valid_F1_history, title = 'F1 score', outputfile = 'training_log/F1_history.png')
        
        data = {"train_loss": train_loss_history, "train_F1": train_F1_history, "valid_loss": valid_loss_history, "valid_F1": valid_F1_history}
        df = pd.DataFrame(data)
        df.to_csv('training_log/training_history.csv')
        

        #Update best model if validation loss is lower than previous best
        if val_loss < best_score:
           best_score = val_loss
           early_stopping_counter = 0
           #Save model
           torch.save(model.state_dict(), model_path)
        else: 
           early_stopping_counter += 1
        
        logging.info(f'This epoch: {val_loss}. Best: {best_score}. Epochs without improvement {early_stopping_counter}.\n')

        #Check early stopping condition
        if early_stopping_counter >= patience: 
            logging.info(f'EARLY STOPPING TRIGGERED: No improvement in {patience} epochs. Stopping training.')
            break
    
    #Save data to csv
    data = {"train_loss": train_loss_history, "train_F1": train_F1_history, "valid_loss": valid_loss_history, "valid_F1": valid_F1_history}
    df = pd.DataFrame(data)
    df.to_csv('training_log/training_history.csv')

    return data

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a model.

    Parameters:
    - model (torch.nn.Module): The model to count the parameters of.

    Returns:
    - int: The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(): 
    argparser = argparse.ArgumentParser(description='train a StructUnet model to predict RNA secondary structure') 
    argparser.add_argument('data', help='directory containing data for training. The directory must contain the following files: train.pkl, valid.pkl', type=str)
    argparser.add_argument('output', help='name of the output file to save the trained model', type=str)
    argparser.add_argument('--epochs', type=int, default=60, help='number of epochs to train the model. (Default = 60)', metavar='')
    argparser.add_argument('--input_dim', type=int, default=8, help='number of channels in the input tensor. Must be either 8 or 16. (Default = 8)', metavar='')
    argparser.add_argument('--hidden_dim', type=int, default=64, help='number of channels in the first hidden layer. (Default = 64)', metavar='')
    argparser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='weight decay for the optimizer. (Default = 0.0)', metavar='')
    argparser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate for the optimizer. (Default = 0.01)', metavar='')
    argparser.add_argument('-p', '--patience', type=int, default=5, help='number of epochs without improvement before stopping training. (Default = 5)', metavar='')
    
    args = argparser.parse_args()

    assert args.input_dim in [8, 16], "ERROR: Invalid input_dim. Must be either 8 or 16"

    os.makedirs('training_log', exist_ok=True)
    logging.basicConfig(filename='training_log/training.log', level=logging.INFO)
    
    train = pickle.load(open(os.path.join(args.data, 'train.pkl'), 'rb'))
    valid = pickle.load(open(os.path.join(args.data, 'valid.pkl'), 'rb'))

    train_dataset = ImageToImageDataset(train)
    valid_dataset = ImageToImageDataset(valid)

    model = RNA_Unet(channels=args.hidden_dim, in_channels=args.input_dim)
    #If model exists, load it and continue training
    if os.path.exists(args.output):
        model.load_state_dict(torch.load(args.output), weights_only=True)
        logging.info(f'\nModel loaded from {args.output}')
    
    logging.info(f"Model has {count_parameters(model)} trainable parameters.")

    fit_model(model, train_dataset, valid_dataset, args.output, patience=args.patience, lr=args.learning_rate, weigth_decay=args.weight_decay, epochs=args.epochs)

if __name__ == '__main__':
    RNA = namedtuple('RNA', 'input output length family name sequence')
    main()