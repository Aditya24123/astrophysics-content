"""Protected-label aggregate diagnostics; never write these into predictor exports."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from hii_sim.io import atomic_json, code_digest, digest, versions


def plot_diagnostics(root, out):
    root=Path(root).resolve(); out=Path(out).resolve()
    if out.is_relative_to(root/'public') or 'public' in out.parts:
        raise ValueError('Label-derived diagnostics must not be written to a public predictor export')
    if out.exists() or out.with_suffix('.json').exists():
        raise ValueError('Choose a new versioned report path; existing diagnostics are preserved')
    plan_path=root/'protected/plan.json'
    plan=json.loads(plan_path.read_text())
    base=[item for item in plan if item['split']!='challenge']
    labels=['log10 EM (pc cm^-6)','Te (K)','log10 major FWHM (arcsec)','axis ratio','PA (deg)']
    keys=['em0_pc_cm6','te_k','major_fwhm_arcsec','axis_ratio','pa_deg']
    values=np.array([[item['source'][key] for key in keys] for item in base],dtype=float)
    ranked=np.column_stack([rankdata(values[:,i]) for i in range(values.shape[1])])
    rank_corr=np.corrcoef(ranked,rowvar=False) if len(base)>1 else None
    plot_values=values.copy(); plot_values[:,0]=np.log10(plot_values[:,0]); plot_values[:,2]=np.log10(plot_values[:,2])
    fig,axes=plt.subplots(2,3,figsize=(12.2,7.4),constrained_layout=True)
    for index,(ax,label) in enumerate(zip(axes.flat[:5],labels)):
        ax.hist(plot_values[:,index],bins=32,color='#245b78',alpha=.86)
        ax.set_xlabel(label); ax.set_ylabel('Independent sources'); ax.grid(alpha=.2)
    ax=axes.flat[5]; counts=Counter(item['split'] for item in base)
    names=['train','validation','test']
    ax.bar(names,[counts[name] for name in names],color=['#4c8c65','#d3a63f','#9a5b62'])
    ax.set_ylabel('Independent physical sources'); ax.set_title('Source-level split counts'); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Protected source-design diagnostic: marginal coverage and grouped splits',fontsize=14)
    out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=150); plt.close(fig)
    diagnostics={'access':'label-derived audit artifact; never a predictor input',
                 'base_sources':len(base),'challenge_sources':len(plan)-len(base),'split_sources':dict(counts),
                 'rank_correlation':rank_corr.tolist() if rank_corr is not None else None,
                 'minima':{key:float(values[:,i].min()) for i,key in enumerate(keys)},
                 'maxima':{key:float(values[:,i].max()) for i,key in enumerate(keys)},
                 'source_plan_sha256':digest(plan_path),'source_provenance_sha256':digest(root/'protected/provenance.json'),
                 'diagnostic_code_sha256':code_digest(),'versions':versions(),'plot':str(out)}
    atomic_json(out.with_suffix('.json'),diagnostics)
    return diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(plot_diagnostics(args.root,args.out),indent=2))


if __name__=='__main__':
    main()
