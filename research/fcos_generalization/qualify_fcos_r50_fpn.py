#!/usr/bin/env python3
"""Append-only parameter-only FCOS CAPP/Width qualification."""
from __future__ import annotations
import argparse, copy, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import nn
try:
    from .fcos_models import (CAPP_PARAMETERS, PARENT_PARAMETERS, WIDTH_PARAMETERS,
        WIDTH_PER_GROUP, GhostSpatial, architecture_hash, capp_model, count_parameters,
        parent_model, set_child, valid_detection_output, width_model)
except ImportError:
    from fcos_models import (CAPP_PARAMETERS, PARENT_PARAMETERS, WIDTH_PARAMETERS,
        WIDTH_PER_GROUP, GhostSpatial, architecture_hash, capp_model, count_parameters,
        parent_model, set_child, valid_detection_output, width_model)

YOLO_PARENT=2_590_035; YOLO_B=1_802_835; YOLO_A=1_432_019

def count(m): return count_parameters(m)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def eligible(model):
    excluded=('backbone.body.conv1','head.classification_head.cls_logits','head.regression_head.bbox_reg','head.regression_head.bbox_ctrness')
    return [(n,m) for n,m in model.named_modules() if isinstance(m,nn.Conv2d) and m.kernel_size==(3,3) and m.groups==1 and not n.startswith(excluded)]
def subset(regions,desired):
    states={0:()}
    for name,saving in regions:
        additions={v+saving:names+(name,) for v,names in states.items()}
        states.update(additions)
        if len(states)>20000:
            states=dict(sorted(states.items(),key=lambda x:(abs(x[0]-desired),len(x[1]),x[1]))[:10000])
    return min(states.items(),key=lambda x:(abs(x[0]-desired),len(x[1]),x[1]))
def valid_output(out):
    return valid_detection_output(out)
def smoke(model,kind,tier,selected,outdir,device):
    model=model.to(device).eval(); image=[torch.zeros(3,640,640,device=device)]
    with torch.inference_mode(): ok1=valid_output(model(image))
    ck=outdir/f'fcos_{kind}_{tier.lower()}_smoke_state.pt'; torch.save(model.state_dict(),ck)
    reload=parent_model() if kind=='parent' else width_model(int(selected)) if kind=='width' else parent_model()
    if kind=='capp':
        for name in selected:
            old=dict(reload.named_modules())[name]; set_child(reload,name,GhostSpatial(old))
    reload.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)); reload=reload.to(device).eval()
    with torch.inference_mode(): ok2=valid_output(reload(image))
    return {'forward_pass':ok1,'save_reload_forward_pass':ok2,'checkpoint_sha256':sha(ck),'checkpoint_bytes':ck.stat().st_size}
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    if not a.launch: raise SystemExit('explicit --launch required')
    if a.output_root.exists(): raise RuntimeError('append-only FCOS qualification output exists')
    a.output_root.mkdir(parents=True); torch.manual_seed(20260903); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parent=parent_model().eval(); pp=count(parent); targets={'B':round(pp*YOLO_B/YOLO_PARENT),'A':round(pp*YOLO_A/YOLO_PARENT)}
    regs=[]
    for name,old in eligible(parent):
        ghost=GhostSpatial(old); saving=sum(p.numel() for p in old.parameters())-sum(p.numel() for p in ghost.parameters())
        if saving>0: regs.append((name,saving,old.in_channels,old.out_channels))
    capp={}
    for tier,target in targets.items():
        saved,names=subset([(r[0],r[1]) for r in regs],pp-target); model=copy.deepcopy(parent)
        for name in names: set_child(model,name,GhostSpatial(dict(model.named_modules())[name]))
        params=count(model)
        if params != CAPP_PARAMETERS[tier]: raise RuntimeError(f'frozen CAPP-{tier} parameter identity changed: {params}')
        capp[tier]={'parameters':params,'target':target,'mismatch_fraction':abs(params-target)/target,'selected_regions':list(names),'parameter_saving':saved,'architecture_hash':architecture_hash(model),'smoke':smoke(model,'capp',tier,names,a.output_root,device)}
        del model; torch.cuda.empty_cache()
    # Fine global search keeps the exact same baseline family. ResNet bottleneck
    # base_width is an integer and imposes no multiple-of-8/32 requirement.
    widths=[]
    for width in range(1,65):
        m=width_model(width); widths.append({'width_per_group':width,'parameters':count(m)}); del m
    width_results={}
    for tier,target in CAPP_PARAMETERS.items():
        best=min(widths,key=lambda r:(abs(r['parameters']-target),r['width_per_group'])); m=width_model(best['width_per_group'])
        if abs(best['parameters']-target)/target > .05:
            raise RuntimeError(f'global Width-{tier} misses frozen CAPP footprint by more than 5%; invoke prospectively defined two-factor fallback')
        if best['width_per_group'] != WIDTH_PER_GROUP[tier] or best['parameters'] != WIDTH_PARAMETERS[tier]:
            raise RuntimeError(f'parameter-only Width-{tier} search result changed')
        width_results[tier]={**best,'target_capp_parameters':target,'mismatch_fraction':abs(best['parameters']-target)/target,
          'architecture_hash':architecture_hash(m),'rounding_rule':'positive integer ResNet width_per_group; fixed 256-channel FPN/head remains GroupNorm-compatible',
          'smoke':smoke(m,'width',tier,best['width_per_group'],a.output_root,device)}
        del m; torch.cuda.empty_cache()
    parent_smoke=smoke(parent,'parent','P','none',a.output_root,device)
    payload={'status':'FCOS_QUALIFICATION_COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),'device':str(device),'parent_parameters':pp,'parent_architecture_hash':architecture_hash(parent),'relative_targets':targets,
      'capp_region_audit':[{'name':n,'saving':s,'in_channels':ci,'out_channels':co} for n,s,ci,co in regs], 'capp':capp,'width_candidates':widths,'width':width_results,'parent_smoke':parent_smoke,
      'width_granularity_audit':{'original_grid':'multiples of 8 in [8,64]','cause':'coarse search grid, not a ResNet/FPN/FCOS legality constraint','resnet_bottleneck_rounding':'int(planes * width_per_group / 64), groups=1; positive integer width_per_group is legal','fpn_coupling':'fixed 256 channels','tower_coupling':'classification and regression towers both fixed at the common 256-channel FPN width','group_norm':'32 groups divide fixed width 256','two_factor_fallback_used':False},
      'production_gate':all(r['mismatch_fraction']<=.05 and r['smoke']['forward_pass'] and r['smoke']['save_reload_forward_pass'] for r in width_results.values()),
      'validation_metric_used':False,'hardware_metric_used':False,'HRSID_TEST_FILES_PRESENT_ON_CLUSTER':False,'HRSID_TEST_METRIC_ACCESS':'CLOSED_AFTER_FINAL_EVALUATION','HARDWARE_AWARE_CAPP_STARTED':False}
    for tier, row in width_results.items():
        atomic(a.output_root/f'FCOS_WIDTH_{tier}_SPEC.json', {'status':'FCOS_WIDTH_SPEC_FROZEN','tier':tier,'family':'single_global_width','width_per_group':row['width_per_group'],'parameters':row['parameters'],'target_capp_parameters':row['target_capp_parameters'],'mismatch_fraction':row['mismatch_fraction'],'architecture_hash':row['architecture_hash'],'rounding_rule':row['rounding_rule'],'validation_metric_used':False,'hardware_metric_used':False,'HRSID_TEST_METRIC_ACCESS':'CLOSED_AFTER_FINAL_EVALUATION','HRSID_TEST_FILES_PRESENT_ON_CLUSTER':False})
    atomic(a.output_root/'FCOS_QUALIFICATION_RESULT.json',payload)
    (a.output_root/'FCOS_QUALIFICATION_REPORT.md').write_text(
      '# FCOS-R50-FPN refined qualification\n\n'
      + f"Parent parameters: {pp}. CAPP B/A mismatch to relative targets: {capp['B']['mismatch_fraction']:.4%}/{capp['A']['mismatch_fraction']:.4%}. "
      + f"Width B/A mismatch to the frozen CAPP footprints: {width_results['B']['mismatch_fraction']:.4%}/{width_results['A']['mismatch_fraction']:.4%}.\n\n"
      + '## Width granularity audit\n\n'
      + 'The original multiples-of-eight grid was unnecessarily coarse. ResNet-50 uses an integer `width_per_group` with groups=1, so every positive integer is legal. '
      + 'The FPN remains fixed at 256 channels; both FCOS towers consume that same width, and GroupNorm(32) therefore remains exactly divisible. '
      + 'No bottleneck, FPN, tower, or GroupNorm constraint caused the earlier miss. The finer parameter-only global search selects '
      + f"Width-B={width_results['B']['width_per_group']} and Width-A={width_results['A']['width_per_group']}. "
      + 'Both pass the <=5% gate, so the prospectively defined two-factor fallback was not used. No loss, AP, validation, or hardware evidence was consulted.\n'
    )
    print(json.dumps({'status':payload['status'],'parent_parameters':pp,'targets':targets,'capp':capp,'width':width_results},indent=2))
if __name__=='__main__': main()
