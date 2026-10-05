"""Append completed evidence to archive refs without changing main or its index."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone,timedelta
import numpy as np
from runtime import ROOT

GIT=r'C:\Users\ykq31\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'


def finalize(progress):
    evidence=[];groups={}
    for item in progress['completed']:
        p=Path(item['result']);r=json.loads(p.read_text(encoding='utf-8'))
        evidence.append({**item,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        if item['stage']=='formal':
            values=[v for e in r['evaluation'] for v in e['returns'].values()]
            groups.setdefault((item['algorithm'],item['track']),[]).append(float(np.mean(values)))
    summary=[]
    for (a,t),values in groups.items():
        assert len(values)==5
        rng=np.random.default_rng(0);samples=np.mean(rng.choice(values,(10000,len(values))),axis=1)
        summary.append({'algorithm':a,'track':t,'seed_count':len(values),'mean':float(np.mean(values)),
                        'seed_std':float(np.std(values,ddof=1)),
                        'bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist()})
    report={'status':'complete','source_sha256':progress['source_sha256'],'cases':evidence,'summary':summary,
            'limitations':'Fixed step budget; no Nash, convergence, privacy or original-paper result claim'}
    encoded=json.dumps(report,indent=2).encode()
    (ROOT/'outputs'/'completed-evidence.json').write_bytes(encoded)
    index=ROOT/'outputs'/'finalize-git.index'
    env={**os.environ,'GIT_INDEX_FILE':str(index),'GIT_AUTHOR_NAME':'RayMARL Maintainer',
         'GIT_AUTHOR_EMAIL':'raymarl@localhost','GIT_COMMITTER_NAME':'RayMARL Maintainer',
         'GIT_COMMITTER_EMAIL':'raymarl@localhost'}
    def git(*args,input=None):
        return subprocess.check_output([GIT,'-C',str(ROOT),*args],input=input,env=env).decode().strip()
    branch='refs/heads/archive/ye2021-maac-validation'
    parent=git('rev-parse',branch)
    git('read-tree',parent)
    blob=git('hash-object','-w','--stdin',input=encoded)
    git('update-index','--add','--cacheinfo',f'100644,{blob},experiments/ye2021_maac/evidence/completed.json')
    tree=git('write-tree')
    commit=git('commit-tree',tree,'-p',parent,'-m','封存完整多种子对照、参数选择与消融证据')
    git('update-ref',branch,commit,parent)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    tag=f'ye2021-validation-complete-{stamp}'
    git('tag',tag,commit)
    (ROOT/'outputs'/'completed-archive.json').write_text(json.dumps({'commit':commit,'tag':tag},indent=2))
    print('COMPLETED_EVIDENCE_ARCHIVED',commit,tag,flush=True)
