import argparse, os, torch, h5py

import pandas as pd
from tqdm import tqdm

from utils.evaluation import compute_sample_metrics, summarize_by_family, summarize_metrics
from utils.model import RNA_Unet, post_process
from utils.plots import plot_F1


def predict(input_tensor: torch.Tensor, model, sequence: str, postprocessing: str, device: str) -> torch.Tensor:
    """
    Run inference and apply the requested post-processing mode.
    """
    output = model(input_tensor.to(device)).squeeze(0).squeeze(0).detach()
    return post_process(output, sequence, postprocessing, device)


def resolve_device(device: str) -> str:
    """
    Resolve the device to use for inference.
    If device is 'auto', it will return 'cuda' if available, otherwise 'cpu'.
    If device is 'cuda' but CUDA is not available, it will raise a ValueError
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict and evaluate RNA secondary structures from a pickled dataset file list.")
    parser.add_argument("dataset", type=str, help="HDF5 file containing the dataset.")
    parser.add_argument("-o", "--output_name", type=str, default="evaluation", help="Name of the output file (extensions will be added automatically).")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the trained model checkpoint (Default: models/RNA_Unet16.pth or models/RNA_Unet8.pth).")
    parser.add_argument("--input_dim", type=int, default=16, help="Number of input channels: 8 or 16.")
    parser.add_argument("--postprocessing", choices=["restricted", "unrestricted", "restricted_noncrossing", "unrestricted_noncrossing"], default="unrestricted", help="Post-processing mode to use for the final structure prediction.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for inference.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Number of hidden channels in the model (Default: 64).")
    args = parser.parse_args()

    if args.input_dim not in [8, 16]:
        raise ValueError("ERROR: Invalid input_dim. Must be either 8 or 16.")

    device = resolve_device(args.device)

    if args.model is None:
        if args.hidden_dim != 64:
            raise ValueError("ERROR: Default pretrained models are only available for hidden_dim=64.")
        args.model = "models/RNA_Unet16.pth" if args.input_dim == 16 else "models/RNA_Unet8.pth"

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print("-- Loading model and dataset --")
    model = RNA_Unet(in_channels=args.input_dim, channels=args.hidden_dim)
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device), weights_only=True))
    model.to(device)

    h5_data = h5py.File(args.dataset, "r")
    files = list(h5_data.keys())
    

    print(f"-- Loaded {len(files)} sequences --")

    rows = []

    with h5py.File(os.path.join('results', f'{args.output_name}_predictions.h5'), "w") as predictions_h5:
        with torch.no_grad():
            for file in tqdm(files, unit="sequence"):
                data = h5_data[file]
                input_tensor = torch.from_numpy(data["input"][()]).unsqueeze(0).type(torch.float32)
                sequence = data["sequence"][()].decode('utf-8')
                predicted = predict(input_tensor, model, sequence, postprocessing=args.postprocessing, device=device)
                group = predictions_h5.create_group(file)
                group.create_dataset("predicted", data=predicted.detach().cpu().numpy(), compression="gzip", compression_opts=4)
                rows.append(compute_sample_metrics(predicted, data, device="cpu"))

    h5_data.close()

    per_sequence_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(per_sequence_df, postprocessing=args.postprocessing, input_dim=args.input_dim)
    family_df = summarize_by_family(per_sequence_df)

    per_sequence_df.to_csv(os.path.join('results', f"{args.output_name}_per_sequence_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join('results', f"{args.output_name}_summary_metrics.csv"), index=False)
    family_df.to_csv(os.path.join('results', f"{args.output_name}_family_metrics.csv"), index=False)

    try:
        plot_F1(per_sequence_df, os.path.join('figures', f"{args.output_name}_per_sequence_F1.png"))
    except Exception as exc:
        print(f"Skipping per-sequence F1 plot due to plotting error: {exc}")

    print("-- Evaluation complete --")


if __name__ == "__main__":
    main()
