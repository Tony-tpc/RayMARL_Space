"""Serial GPU matrix with resumable case records and explicit completion state."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import traceback
import time
import torch
import ray
from run import run_case
from runtime import ROOT, initialize_ray, stop_owned_ray


def code_hash():
    digest=hashlib.sha256()
    for f in sorted((ROOT/"src"/"raymarl").rglob("*.py")):
        digest.update(f.relative_to(ROOT).as_posix().encode());digest.update(f.read_bytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stages",nargs="+",choices=("smoke","debug","formal"),default=["smoke"])
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    destination=ROOT/"outputs"/"suite-progress.json"
    specifications={"smoke":(range(1),1000),"debug":(range(3),5000),"formal":(range(5),50000)}
    cases=[]
    for stage in args.stages:
        seeds,budget=specifications[stage]
        for track in ("continuous","hybrid"):
            algorithms=("maac","isac","masac","maddpg","mappo","ippo") if track=="continuous" else (
                "maac","isac","masac","mappo","ippo")
            cases += [(stage,algorithm,track,seed,budget) for algorithm in algorithms for seed in seeds]
    progress={"status":"running","source_sha256":code_hash(),"total_cases":len(cases),"completed":[],
              "started_unix":time.time(),"stages":args.stages}
    destination.parent.mkdir(exist_ok=True)
    stop_owned_ray()
    initialize_ray()
    try:
        for stage,algorithm,track,seed,budget in cases:
            progress["current"]={"stage":stage,"algorithm":algorithm,"track":track,"seed":seed,
                                  "requested_joint_steps":budget}
            destination.write_text(json.dumps(progress,indent=2),encoding="utf-8")
            result_path=ROOT/"outputs"/stage/f"{algorithm}-{track}-seed{seed}"/"result.json"
            if args.resume and result_path.exists():
                result=json.loads(result_path.read_text())
                if result["requested_joint_env_steps"]!=budget:
                    raise ValueError(f"Resume budget mismatch: {result_path}")
            else:
                result=run_case(algorithm,track,seed,budget,stage,1,168,
                                evaluate_test=stage in ("smoke","formal"))
            progress["completed"].append({**progress["current"],"result":str(result_path),
                "actual_joint_steps":result["actual_joint_env_steps"]})
            gc.collect();torch.cuda.empty_cache()
        progress["status"]="complete"
    except Exception:
        progress["status"]="failed";progress["error"]=traceback.format_exc()
        raise
    finally:
        progress["finished_unix"]=time.time()
        destination.write_text(json.dumps(progress,indent=2),encoding="utf-8")
        ray.shutdown()


if __name__=="__main__":
    main()
