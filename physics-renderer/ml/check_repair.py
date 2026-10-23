"""Regression launcher with explicit, fresh fixture directory; no physics edits."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from hii_sim.io import atomic_json

FILES=['tests/test_npe_contract.py','tests/test_training_data.py','ml/test_ml.py','ml/test_followup.py','ml/test_review_v2.py','ml/test_resume_conditionals.py','ml/test_probit_probe.py','ml/test_angle_refresh.py','ml/test_model_information.py','ml/test_repair_campaign.py','ml/test_repair_refresh.py','ml/test_repair_resume.py']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fixture-root',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output/ml_initial_20261023/repair_02'));a=ap.parse_args()
    if a.fixture_root.exists():raise FileExistsError('Choose a fresh fixture directory; launcher never deletes fixtures')
    if Path('output/ml_initial_20261023/continuation.lock').exists():raise RuntimeError('Scientific job active; keep work serialized')
    a.fixture_root.parent.mkdir(parents=True,exist_ok=True);a.out.mkdir(exist_ok=True)
    command=[sys.executable,'-B','-m','pytest','-q',*FILES,'-p','no:cacheprovider','--basetemp',str(a.fixture_root.resolve())]
    start=time.perf_counter();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONPATH='src;ml',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');r=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8')
    (a.out/'regressions.log').write_text(r.stdout+r.stderr,encoding='utf-8');print(r.stdout+r.stderr,flush=True)
    atomic_json(a.out/'tests_receipt.json',dict(utc=datetime.now(timezone.utc).isoformat(),command=command,exit_code=r.returncode,passed=r.returncode==0,wall_seconds=time.perf_counter()-start,fixture_root=str(a.fixture_root.resolve()),fixture_policy='Fresh explicitly chosen path, no cleanup/deletion, no physics/source-I/O changes, no automatic failure suppression. A numerical failure is never retried as an environment failure.',log='regressions.log'))
    sys.exit(r.returncode)

if __name__=='__main__':main()
