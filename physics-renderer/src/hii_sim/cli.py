"""Command line entry points for bounded, reproducible phase-one runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

from .dataset import generate, make_plan
from .casa_bridge import prepare_casa_inputs
from .schema import load_config
from .validation import validate
from .config_defaults import DEFAULT_CONFIG, LEGACY_CONFIG


def build_parser():
    parser = argparse.ArgumentParser(prog="hii-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="Generate or resume a bounded dataset")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Configuration file (default: {DEFAULT_CONFIG}; use {LEGACY_CONFIG} only for explicit legacy replay)")
    p.add_argument("--out", required=True)
    p.add_argument("--sources", type=int, required=True, help="Independent base physical sources")
    p.add_argument("--limit", type=int, help="Maximum new physical sources in this invocation")
    p.add_argument("--max-seconds", type=float, default=1800)
    p.add_argument("--max-gib", type=float, default=20)
    p.add_argument("--source-headroom-seconds",type=float,default=0.,help="Measured one-parent/finalization time reservation; cooperative, not hard preemption")
    p.add_argument("--source-headroom-mib",type=float,default=0.,help="One-parent/finalization disk reservation in MiB")
    p = sub.add_parser("plan", help="Report parameter coverage without rendering images")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Configuration file (default: {DEFAULT_CONFIG}; use {LEGACY_CONFIG} only for explicit legacy replay)")
    p.add_argument("--sources", type=int, required=True)
    p = sub.add_parser("validate", help="Run physics, gridding, and sensitivity checks")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Configuration file (default: {DEFAULT_CONFIG}; use {LEGACY_CONFIG} only for explicit legacy replay)")
    p.add_argument("--out", default="output/validation")
    p = sub.add_parser("prepare-casa", help="Convert one protected truth cube to CASA input planes")
    p.add_argument("--truth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42017)
    p.add_argument("--duration", default="30min")
    p.add_argument("--integration", default="10s")
    p.add_argument("--hourangle", default="transit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        result = generate(args.out, args.config, args.sources, args.limit,
                          args.max_seconds, args.max_gib,
                          source_headroom_seconds=args.source_headroom_seconds,source_headroom_mib=args.source_headroom_mib)
    elif args.command == "plan":
        config = load_config(args.config)
        plan = make_plan(config, args.sources)
        base = [x for x in plan if x["split"] != "challenge"]
        keys = list(config["sampling"]["ranges"])
        x = np.array([[item["source"][k] for k in keys] for item in base])
        result = {"base_sources": args.sources, "challenge_sources": config["challenge_sources"],
                  "realizations_per_source": config["realizations_per_source"],
                  "total_realizations": len(plan)*config["realizations_per_source"],
                  "splits": {s: sum(item["split"] == s for item in plan)
                             for s in ("train", "validation", "test", "challenge")},
                  "base_parameter_min_max": {k:[float(x[:,i].min()),float(x[:,i].max())]
                                               for i,k in enumerate(keys)},
                  "rank_correlation": np.corrcoef(rankdata(x,axis=0),rowvar=False).tolist() if len(base)>1 else None,
                  "target_list": config["reported_targets"]}
    elif args.command == "validate":
        result = validate(args.config, args.out)
    else:
        result = prepare_casa_inputs(args.truth, args.out, args.seed,
                                     args.duration, args.integration, args.hourangle)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("status")=="budget_exceeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
