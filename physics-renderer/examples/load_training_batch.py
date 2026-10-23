"""Print one verified training batch; no optimizer, model or training."""
import argparse
from pathlib import Path
from hii_sim.training_data import iter_batches, load_transform


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--prepared",required=True,type=Path)
    ap.add_argument("--batch-size",type=int,default=32)
    args = ap.parse_args()
    transform = load_transform(args.prepared,args.root/"public")
    batch = next(iter_batches(args.root,transform,split="train",batch_size=args.batch_size,
                              shuffle=True,order_seed=22022,include_targets=True))
    print("Model input:",batch["features"]["values"].shape,batch["features"]["values"].dtype,"dimensionless")
    print("Original targets:",batch["targets"]["original"].shape,batch["targets"]["order"])
    print("Original target units:",batch["targets"]["units"])
    print("Target transform: log10 EM0/size; identity Te/q; then train-parent z scores")
    print("Observing thermal covariance: Jy^2 before input scaling; taper is not inverse variance")
    print("Bookkeeping is outside features. No training executed.")


if __name__ == "__main__":
    main()
