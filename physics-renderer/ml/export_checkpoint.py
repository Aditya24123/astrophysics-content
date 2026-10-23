"""Prepare an allowlisted text-only GitHub checkpoint, never dataset artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    args = ap.parse_args()
    paths = []
    for directory in ('src/hii_sim', 'tests', 'tools', 'configs', 'examples', 'ml'):
        paths.extend(p for p in Path(directory).glob('*') if p.is_file() and p.suffix in ('.py', '.json', '.md'))
    paths.extend(p for p in Path('docs').glob('*.md') if p.name != 'CASA_SETUP_LOG_2026-10-20.md')
    paths.extend(Path(n) for n in ('README.md', 'pyproject.toml', 'requirements.txt', 'requirements-dev.txt', '.gitignore'))
    paths.append(args.summary)
    entries, manifest, findings = [], [], []
    for path in sorted(set(paths)):
        content = path.read_text(encoding='utf-8')
        if re.search(r'(?i)(gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)', content):
            findings.append(str(path))
        if re.search(r'(?i)[A-Z]:[\\/]+Users[\\/]|/home/[a-zA-Z0-9_-]+/', content):
            raise ValueError('Machine-specific personal path in allowlisted file: '+str(path))
        rel = path.as_posix()
        if path == args.summary:
            rel = 'CHECKPOINT_STATUS.md'
        target = 'physics-renderer/'+rel
        entries.append({'path': target, 'mode': '100644', 'type': 'blob', 'content': content})
        manifest.append({'path': target, 'bytes_utf8': len(content.encode()),
                         'sha256': hashlib.sha256(content.encode()).hexdigest()})
    if findings:
        raise ValueError('Secret pattern in '+str(findings))
    # All generated banks, labels, truth, logs, local environment and credentials
    # are excluded by this explicit file allowlist; no broad add operation.
    record = {'files': manifest, 'secret_findings': findings,
              'excluded': ['output datasets and protected labels/truth', '.venv', 'caches', 'logs',
                           '.git', 'credentials and environment files', 'machine-specific CASA setup log'],
              'visibility': 'public', 'destination': 'Aditya24123/astrophysics-content/physics-renderer'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'entries': entries, 'scan': record}), encoding='utf-8')
    args.out.with_suffix('.scan.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'files': len(entries), 'bytes_utf8': sum(x['bytes_utf8'] for x in manifest),
                      'secret_findings': findings, 'scan': str(args.out.with_suffix('.scan.json'))}))


if __name__ == '__main__':
    main()
