import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PATHS
from src.utils.io_utils import load_features, load_json
from src.baselines.fixed_window import fixed_window_segments
from src.baselines.shot_detection import shot_segments_from_visual_signal
from src.stage1_boundary.inference import load_stage1_model, segment_video
from src.stage1_boundary.pseudo_labels import pseudo_boundaries_from_signals, boundaries_from_annotations
from src.stage2_retrieval.retrieval import LectureRetriever
from src.baselines.sliding_window_retrieval import build_sliding_window_index, search_sliding_window

OUT = ROOT / "output" / "evaluation_suite"

def score_boundaries(pred, gold, step, tol_sec):
    p, g = sorted(pred), sorted(gold)
    used=set(); tp=0
    for x in p:
        candidates=[(abs(x-y),i) for i,y in enumerate(g) if i not in used and abs(x-y)<=tol_sec]
        if candidates:
            _,i=min(candidates); used.add(i); tp+=1
    precision=tp/len(p) if p else (1.0 if not g else 0.0)
    recall=tp/len(g) if g else (1.0 if not p else 0.0)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    # Boundary arrays at sampled steps; Pk/WD compare induced segment IDs.
    lo=min([*p,*g], default=0); hi=max([*p,*g], default=0)
    n=max(2, int(math.ceil((hi-lo)/step))+2)
    pb=np.zeros(n-1,dtype=int); gb=np.zeros(n-1,dtype=int)
    for x in p: pb[min(n-2,max(0,round((x-lo)/step)))]=1
    for x in g: gb[min(n-2,max(0,round((x-lo)/step)))]=1
    def labels(b): return np.r_[0,np.cumsum(b)]
    pl,gl=labels(pb),labels(gb)
    k=max(1,round(np.mean(np.diff(np.r_[0,np.where(gb)[0]+1,n]))/2))
    pk=wd=0; denom=max(1,n-k)
    for i in range(denom):
        pk += int((pl[i]==pl[i+k]) != (gl[i]==gl[i+k]))
        wd += int(np.sum(pb[i:i+k]) != np.sum(gb[i:i+k]))
    return {"precision":precision,"recall":recall,"f1":f1,"pk":pk/denom,"windowdiff":wd/denom}

def internal_boundaries(segments): return [float(s["end_time"]) for s in segments[:-1]]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={"project_root":str(ROOT),"warnings":[],"validation":[],"table_ii":{},"table_iii":{"status":"pending","queries":0,"iou_hit_threshold":0.3}}
    videos={p.stem:p for p in Path(PATHS.raw_videos_dir).glob("*") if p.is_file()}
    features={p.stem:p for p in Path(PATHS.features_dir).glob("*.npz")}
    annotations=Path(PATHS.annotations_dir)
    boundary_files=sorted(annotations.glob("*_boundaries.json"))
    query_files=sorted(annotations.glob("*_queries.json"))
    report["inventory"]={"videos":len(videos),"feature_files":len(features),"boundary_files":len(boundary_files),"query_files":len(query_files)}
    for f in boundary_files:
        vid=f.name[:-len("_boundaries.json")]
        try:
            a=load_json(str(f)); ts=a["boundary_times_sec"]
            if vid not in videos or vid not in features: raise ValueError("matching video or feature file missing")
            if any(not isinstance(x,(int,float)) for x in ts) or ts!=sorted(set(ts)): raise ValueError("boundary timestamps must be unique, numeric, ascending seconds")
            feat=load_features(str(features[vid])); times=feat["timestamps"]; step=float(np.median(np.diff(times)))
            start=float(a.get("teaching_start_sec",times[0])); end=float(a.get("teaching_end_sec",times[-1]+step))
            if not times[0] <= start < end <= times[-1]+step: raise ValueError("teaching span outside feature timeline")
            if any(not start < x < end for x in ts): raise ValueError("boundary outside teaching span")
            report["validation"].append({"file":f.name,"status":"ok","boundaries":len(ts)})
        except Exception as e: report["validation"].append({"file":f.name,"status":"error","detail":str(e)}); continue
        gold=np.asarray(ts,float); eval_rows=[]
        methods={"fixed_60":internal_boundaries(fixed_window_segments(times,60)),"fixed_120":internal_boundaries(fixed_window_segments(times,120)),"shot_change":internal_boundaries(shot_segments_from_visual_signal(times,feat["visual_signal"]))}
        pseudo=pseudo_boundaries_from_signals(feat["ocr_signal"],feat["topic_drift_signal"],feat["visual_signal"])
        methods["pseudo_label_peaks"]=[float(times[min(i+1,len(times)-1)]) for i in np.where(pseudo==1)[0]]
        ck=Path(PATHS.checkpoints_dir)/"stage1_boundary_detector.pt"
        if ck.exists():
            model=load_stage1_model(str(ck),"cuda" if __import__("torch").cuda.is_available() else "cpu")
            seg=segment_video(vid,model,PATHS.features_dir,"cuda" if __import__("torch").cuda.is_available() else "cpu")
            methods["conceptbound_stage1"]=[float(x) for x in internal_boundaries(seg)]
        for name,pred in methods.items():
            row={"video_id":vid,"method":name,"segments":len(pred)+1}
            for tol in (15,30): row[f"f1@{tol}s"]=score_boundaries(pred,gold,step,tol)["f1"]
            row.update({k:v for k,v in score_boundaries(pred,gold,step,30).items() if k in ("pk","windowdiff")})
            eval_rows.append(row)
        report["table_ii"][vid]=eval_rows
    query_count=0
    for f in query_files:
        vid=f.name[:-len("_queries.json")]
        try:
            a=load_json(str(f)); pairs=a["pairs"]
            if vid not in videos: raise ValueError("matching video missing")
            for i,p in enumerate(pairs):
                q=p["query"]; s=float(p["start_time"]); e=float(p["end_time"])
                if not isinstance(q,str) or len(q.strip())<8 or not 0<=s<e: raise ValueError(f"invalid pair at index {i}")
            query_count+=len(pairs); report["validation"].append({"file":f.name,"status":"ok","queries":len(pairs),"note":"human correctness requires review"})
        except Exception as e: report["validation"].append({"file":f.name,"status":"error","detail":str(e)})
    report["table_iii"]["queries"]=query_count
    # Macro aggregate by lecture for available segmentation measures.
    flat=[r for rows in report["table_ii"].values() for r in rows]
    agg={}
    for name in sorted(set(r["method"] for r in flat)):
        rows=[r for r in flat if r["method"]==name]
        agg[name]={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("video_id","method")}
        agg[name]["lectures"]=len(rows)
    report["table_ii_macro"]=agg
    # Retrieval evaluation using the two retrieval implementations currently in the project.
    # Rank against the complete course corpus, as required for multi-video search.
    queries=[]
    for f in query_files:
        vid=f.name[:-len("_queries.json")]
        try:
            for p in load_json(str(f))["pairs"]:
                queries.append({"video_id":vid,"query":p["query"],"start_time":float(p["start_time"]),"end_time":float(p["end_time"])})
        except Exception: pass
    def retrieval_metrics(ranked):
        if not ranked: return {"R@1":None,"R@5":None,"MRR":None,"mIoU":None}
        hits1=[]; hits5=[]; rr=[]; ious=[]
        for results,q in zip(ranked,queries):
            def iou(r):
                if r.get("video_id")!=q["video_id"]: return 0.0
                inter=max(0.0,min(r["end_time"],q["end_time"])-max(r["start_time"],q["start_time"]))
                union=max(r["end_time"],q["end_time"])-min(r["start_time"],q["start_time"])
                return inter/union if union>0 else 0.0
            vals=[iou(r) for r in results]
            ranks=[i+1 for i,v in enumerate(vals) if v>=0.3]
            hits1.append(int(bool(ranks and ranks[0]<=1))); hits5.append(int(bool(ranks and ranks[0]<=5)))
            rr.append(1/ranks[0] if ranks else 0.0); ious.append(vals[0] if vals else 0.0)
        return {"R@1":float(np.mean(hits1)),"R@5":float(np.mean(hits5)),"MRR":float(np.mean(rr)),"mIoU":float(np.mean(ious))}
    test_rows=[]
    if queries:
        dev="cuda" if __import__("torch").cuda.is_available() else "cpu"
        try:
            retriever=LectureRetriever(PATHS.index_dir,Path(PATHS.checkpoints_dir)/"stage2_retrieval_model.pt",dev)
            main_ranked=[retriever.search(q["query"],top_k=20) for q in queries]
            test_rows.append({"method":"cross_modal_retrieval_existing_index","metrics":retrieval_metrics(main_ranked)})
        except Exception as e:
            report["warnings"].append("Cross-modal retrieval unavailable: "+str(e))
        try:
            vids=sorted(set(q["video_id"] for q in queries))
            emb,meta=build_sliding_window_index(vids,features_dir=PATHS.features_dir,device=dev)
            sw_ranked=[search_sliding_window(q["query"],emb,meta,top_k=20,device=dev) for q in queries]
            test_rows.append({"method":"sliding_window_existing_baseline","metrics":retrieval_metrics(sw_ranked)})
        except Exception as e:
            report["warnings"].append("Sliding-window retrieval unavailable: "+str(e))
    report["table_iii"]={"status":"computed for available project retrievers; other paper rows are not implemented","queries":len(queries),"hit_iou_threshold":0.3,"results":test_rows,"unavailable_rows":["BM25","BGE","RRF + rerank","ground-truth segmentation + RRF + rerank"],"caveat":"Scores are descriptive if these same queries were used to train Stage 2; use a frozen held-out query set for paper claims."}
    (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (OUT/"table_ii_per_lecture.csv").open("w",newline="",encoding="utf-8") as f:
        rows=[r for vv in report["table_ii"].values() for r in vv]
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT/"table_iii.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["method","R@1","R@5","MRR","mIoU"]); w.writeheader()
        for item in test_rows: w.writerow({"method":item["method"],**item["metrics"]})
    print(f"Report written to {OUT / 'report.json'}")
    print(f"Validated {len(report['validation'])} annotation files; counted {query_count} queries.")
    print("Table III retrieval scores remain unavailable until its retrieval baselines are implemented and run.")

if __name__=="__main__": main()
