import argparse, os, pickle, torch
from collections import namedtuple
from tqdm import tqdm

from utils.model import RNA_Unet, blossom_postprocessing, prepare_postprocessing

RNA = namedtuple("RNA", "input output length family name sequence")


def predict(input_tensor: torch.Tensor, model, sequence: str, postprocessing: str, device: str) -> torch.Tensor:
    output = model(input_tensor.to(device)).squeeze(0).squeeze(0).detach()
    if postprocessing == "restricted":
        output = prepare_postprocessing(output, sequence, device)
    else:
        output = (output + output.T) / 2
    return blossom_postprocessing(output, sequence, device)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict RNA secondary structures for an entire pickled dataset file list.")
    parser.add_argument("dataset", type=str, help="Pickle file containing a list of per-sample pickle files.")
    parser.add_argument("-o", "--output", type=str, default="results/predictions", help="Output directory.")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the trained model checkpoint (Default: models/RNA_Unet16.pth or models/RNA_Unet8.pth).")
    parser.add_argument("--input_dim", type=int, default=16, help="Number of input channels: 8 or 16.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Number of channels in the first hidden layer.")
    parser.add_argument("--postprocessing", choices=["restricted", "unrestricted"], default="unrestricted", help="Post-processing mode to use for the final structure prediction.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for inference.")
    args = parser.parse_args()

    if args.input_dim not in [8, 16]:
        raise ValueError("ERROR: Invalid input_dim. Must be either 8 or 16.")

    device = resolve_device(args.device)

    if args.model is None:
        if args.hidden_dim != 64:
            raise ValueError("ERROR: Default pretrained models are only available for hidden_dim=64.")
        args.model = "models/RNA_Unet16.pth" if args.input_dim == 16 else "models/RNA_Unet8.pth"

    os.makedirs(args.output, exist_ok=True)

    print("-- Loading model and dataset --")
    model = RNA_Unet(in_channels=args.input_dim, channels=args.hidden_dim)
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device), weights_only=True))
    model.to(device)
    model.eval()

    files = pickle.load(open(args.dataset, "rb"))
    print(f"-- Loaded {len(files)} sequences --")

    progress_bar = tqdm(total=len(files), unit="sequence")
    with torch.no_grad():
        for file in files:
            data = pickle.load(open(file, "rb"))
            basename = os.path.basename(file)
            predicted = predict(data.input.unsqueeze(0), model, data.sequence, postprocessing=args.postprocessing, device=device)
            pickle.dump(predicted, open(os.path.join(args.output, basename), "wb"))
            progress_bar.update(1)
    progress_bar.close()

    print("-- Predictions complete --")


if __name__ == "__main__":
    main()
