import argparse, torch, sys, os

from time import time

from utils.data import read_fasta, make_matrix_from_sequence_8, make_matrix_from_sequence_16, write_bpseq, write_ct, write_to_stdout
from utils.model import RNA_Unet, blossom_postprocessing, prepare_postprocessing

def predict(input: torch.Tensor, model, sequence: str, postprocessing: str = 'unrestricted', device: str = 'cpu'):
    """
    Predict the RNA 2D structure for a given input sequence from the tensor representation.

    Parameters:
    - input (torch.Tensor): The input tensor.
    - model: The trained model.
    - sequence (str): The input RNA sequence.
    - postprocessing (str): OPTIONAL. Either 'restricted' or 'unrestricted'. Default is 'unrestricted'.
    - device (str): OPTIONAL. The device to use for the computation. Default is 'cpu'.

    Returns:
    - tensor: The predicted RNA 2D structure.
    """
    output = model(input.to(device)).squeeze(0).squeeze(0).detach()
    if postprocessing == 'restricted':
        output = prepare_postprocessing(output, sequence, device)
    else:
        output = (output + output.T)/2

    return blossom_postprocessing(output, sequence, device)

def main(): 
    argparser = argparse.ArgumentParser(description='predict RNA secondary structure from a sequence using StructUnet')
    argparser.add_argument('-i', '--input', type=str, help='input RNA sequence. Can be a sequence or a path to a fasta file', metavar='')
    argparser.add_argument('-o', '--output', type=str, help='path to save the predicted RNA 2D structure. Valid file formats are .ct and .bpseq (Default = stdout)', default=sys.stdout, metavar='')
    argparser.add_argument('-m', '--model', type=str, help='path to the trained model. (Default is models/RNA_Unet8.pth or models/RNA_Unet16.pth dependent on the input_dim)', default=None, metavar='')
    argparser.add_argument('--input_dim', type=int, help='number of input channels. (Default = 16)', default=16, metavar='')
    argparser.add_argument('--hidden_dim', type=int, help='number of hidden channels. (Default = 64)', default=64, metavar='')
    argparser.add_argument('--postprocessing', choices=['restricted', 'unrestricted'], default='unrestricted', help="post-processing mode for final predictions. (Default = unrestricted)", metavar='')
    argparser.add_argument('-std', '--only_standardbp', action='store_true', help='deprecated alias for --postprocessing restricted')
    
    args = argparser.parse_args()

    #Check arguments
    assert args.input_dim in [8, 16], "ERROR: Invalid input_dim. Must be either 8 or 16"

    if not args.input:
        raise ValueError('ERROR: No input sequence provided. Provide an input sequence with the -i flag as a string or a path to a fasta file')

    if args.only_standardbp:
        args.postprocessing = 'restricted'
    
    if os.path.isfile(args.input):
        sequences = read_fasta(args.input)
    else:
        sequences = [(args.input, 'User inputted sequence')]

    #Set up output format
    if args.output != sys.stdout:
        output_map = {'.bpseq': write_bpseq, '.ct': write_ct}
        file_type = os.path.splitext(args.output)[1]
        assert file_type in output_map, f"ERROR: Invalid output file format. Must be one of {list(output_map.keys())}"
        to_outputfile = output_map[file_type]
    else:
        to_outputfile = write_to_stdout

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Finding deault models
    if args.model is None:
        args.model = 'models/RNA_Unet16.pth' if args.input_dim == 16 else 'models/RNA_Unet8.pth'
        
        if args.hidden_dim != 64:
            #Make sure that the hidden_dim is 32, to match the pretrained models
            raise ValueError('ERROR: No pretrained model available for hidden_dim other than 32')

    
    print('-- Loading model --')
    model = RNA_Unet(in_channels=args.input_dim, channels=args.hidden_dim)
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device), weights_only=True))
    model.to(device)
    model.eval()


    print('-- Predicting RNA 2D structure --')
    start_time = time()

    for i, (sequence, seq_name) in enumerate(sequences):
        if args.input_dim == 8:
            input_matrix = make_matrix_from_sequence_8(sequence).to(device)
        elif args.input_dim == 16:
            input_matrix = make_matrix_from_sequence_16(sequence)
        
        output = predict(input_matrix.unsqueeze(0), model, sequence, args.postprocessing, device)

        #Add index to output file if multiple sequences are predicted to prevent overwriting
        if i>0 and args.output != sys.stdout: 
            output_file = args.output.replace(f'{file_type}', f'_{i}{file_type}')
        else:
            output_file = args.output
        
        to_outputfile(output_file, sequence, output, seq_name)
    
    print(f'-- {len(sequences)} predictions done in {(time()-start_time):.2f} seconds --')



if __name__ == '__main__':
    main()
