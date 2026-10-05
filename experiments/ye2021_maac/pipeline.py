"""Complete serial experiment plan; source/result hashes guard resumption."""
import gc
import itertools
import json
import time
import traceback
import torch
import ray
from run import run_case
from runtime import ROOT,initialize_ray,stop_owned_ray
from suite import code_hash


def main():
    progress={'status':'running','source_sha256':code_hash(),'completed':[],'started_unix':time.time()}
    path=ROOT/'outputs'/'pipeline-progress.json'
    stop_owned_ray();initialize_ray()
    selected={}
    try:
        configurations=[]
        for track in ('continuous','hybrid'):
            algorithms=['maac','isac','masac','mappo','ippo','cen_sac']
            if track=='continuous':algorithms.append('maddpg')
            configurations += [(track,a) for a in algorithms]
        def execute(stage,a,track,seed,budget,params=None,label=None):
            key=label or f'{a}-{track}-seed{seed}'
            progress['current']={'stage':stage,'algorithm':a,'track':track,'seed':seed,'budget':budget,'case':key}
            path.write_text(json.dumps(progress,indent=2),encoding='utf-8')
            result_path=ROOT/'outputs'/stage/key/'result.json'
            if result_path.exists():
                r=json.loads(result_path.read_text())
                if r['requested_joint_env_steps']!=budget or r.get('parameters',{})!=(params or {}) or r.get('source_sha256')!=code_hash():
                    raise ValueError(f'Resume signature mismatch: {key}')
            else:
                r=run_case(a,track,seed,budget,stage,1,168,stage in ('formal','smoke','ablation'),params,label)
            progress['completed'].append({**progress['current'],'result':str(result_path),
                                          'actual_steps':r['actual_joint_env_steps']})
            gc.collect();torch.cuda.empty_cache()
            return r
        for track,a in configurations:
            for seed in range(3):execute('debug',a,track,seed,5000)
        for track,a in configurations:
            best=None
            for index,(lr,second) in enumerate(itertools.product((1e-4,3e-4,1e-3),(0,1))):
                params={'actor_lr':lr}
                if a in ('mappo','ippo'):params['entropy_coeff']=(0.,.01)[second]
                elif a=='maddpg':params['exploration_std']=(.1,.2)[second]
                else:params['temperature']=(.05,.1)[second]
                r=execute('tune',a,track,0,10000,params,f'{a}-{track}-candidate{index}')
                score=r['validation_best_mean_personal_return']
                if best is None or score>best[0]:best=(score,params)
            selected[f'{a}/{track}']=best[1]
        (ROOT/'outputs'/'selected-configs.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
        for track,a in configurations:
            for seed in range(5):execute('formal',a,track,seed,50000,selected[f'{a}/{track}'])
        for a in ('maac_uniform','maac_no_baseline','maac_no_history'):
            params=dict(selected['maac/hybrid'])
            if a=='maac_uniform':params['critic_mode']='uniform'
            if a=='maac_no_baseline':params['counterfactual']=False
            for seed in range(3):execute('ablation',a,'hybrid',seed,50000,params)
        progress['status']='complete'
        from finalize import finalize
        finalize(progress)
    except Exception:
        progress['status']='failed';progress['error']=traceback.format_exc();raise
    finally:
        progress['finished_unix']=time.time();path.write_text(json.dumps(progress,indent=2),encoding='utf-8')
        ray.shutdown()


if __name__=='__main__':main()
