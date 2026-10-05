"""Build a compact manifest; generated data is never staged in main."""
import hashlib
import json
import subprocess
from pathlib import Path
from runtime import ROOT
from suite import code_hash


def capture():
    data=ROOT/'datasets'/'citylearn'/'datasets'/'citylearn_challenge_2022_phase_all_plus_evs'
    files=[{'path':f.relative_to(data).as_posix(),'bytes':f.stat().st_size,
            'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(data.rglob('*')) if f.is_file()]
    requirements={}
    for name in ('.venv','.venv-citylearn'):
        requirements[name]=subprocess.check_output([str(ROOT/name/'Scripts'/'python.exe'),'-m','pip','freeze'],text=True)
    results=[]
    for f in sorted((ROOT/'outputs'/'smoke').glob('*/result.json')):
        r=json.loads(f.read_text());results.append({'case':f.parent.name,'actual_steps':r['actual_joint_env_steps'],
            'source_sha256':r.get('source_sha256'),'result_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
            'elapsed_seconds':r['elapsed_seconds']})
    preflight=json.loads((ROOT/'outputs'/'preflight.json').read_text())
    record={'source_sha256':code_hash(),'citylearn':'3.0.2','dataset_tag':'v3.0.2','data_files':files,
            'dependency_snapshots':requirements,'game_qualification':preflight,'smoke_cases':results,
            'formal_status':'not_complete'}
    out=ROOT/'experiments'/'ye2021_maac'/'evidence';out.mkdir(exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


if __name__=='__main__':capture()
