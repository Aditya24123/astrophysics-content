"""Build a separate v2 export; never repair, unlock or rewrite the source in place.

Usage: python tools/rebuild_dataset_index.py --source DATASET --out NEW_EXPORT
This copies verified products and can require as much disk as the source dataset.
"""
import argparse
import json
from pathlib import Path
from hii_sim.export import export_v2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_v2(args.source,args.out),indent=2))


if __name__=="__main__":
    main()
