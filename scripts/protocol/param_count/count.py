"""Numerical implementation used by the reported experiment."""
from torch.utils._python_dispatch import TorchDispatchMode
import inspect
import torch
import traceback
from torch.utils._pytree import tree_flatten


def reference(filename, function):
    return dict(module=getattr(function,'__module__',None),name=getattr(function,'__qualname__',None))

def group(name):
    parts=name.split('.')
    if name=='weight_token': return 'fusion'
    if name.startswith(('frame_branch_query_','text_weight_fc')): return 'query'
    if name.startswith(('query_','modular_vector_mapping','textual_perturbation_sampler')): return 'query'
    if name.startswith(('clip_','mean_clip_encoder')):
        if parts[0]=='clip_encoder' and parts[1] in ('ca','layer1','layer2'): return 'fusion'
        return 'clip'
    if name.startswith('frame_'):
        if name.startswith('frame_module.'):
            return 'query' if any(x in name for x in ('query_encoder','query_pos_embed','query_slot_attention')) else 'frame'
        if parts[0] in ('frame_encoder','frame_encoder_1') and parts[1] in ('ca','layer1','layer2'): return 'fusion'
        return 'frame'
    return 'other'


def census(model,case):
    params=dict(model.named_parameters())  # PyTorch removes repeated parameter objects.
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False): aliases.setdefault(id(p),[]).append(name)
    pid={id(p):name for name,p in params.items()}
    storage={p.untyped_storage().data_ptr():id(p) for p in params.values()}
    called={}; used=set(); used_ops={}; handles=[]
    module_ref={}
    for name,m in model.named_modules():
        def hook(mod,args,out,n=name): called[n]=called.get(n,0)+1
        handles.append(m.register_forward_hook(hook))
        if list(m.parameters(recurse=False)):
            try: module_ref[name]=reference(inspect.getsourcefile(type(m)),type(m).forward)
            except (TypeError,OSError): module_ref[name]={'class':str(type(m))}
    class Record(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            for t in tree_flatten((args,kwargs))[0]:
                if isinstance(t,torch.Tensor) and t.device.type=='cpu':
                    key=id(t) if id(t) in pid else storage.get(t.untyped_storage().data_ptr())
                    if key in pid:
                        used.add(key); used_ops.setdefault(pid[key],set()).add(str(func))
            return func(*args,**(kwargs or {}))
    torch.manual_seed(20261031)
    model.eval()
    d=model.config.visual_input_size; qd=model.config.query_input_size
    c=torch.randn(2,model.config.map_size,d)
    f=torch.randn(2,model.config.max_ctx_l,d)
    fm=torch.ones(2,model.config.max_ctx_l)
    q=torch.randn(2,7,qd); qm=torch.ones(2,7)
    case['synthetic_inputs']=dict(clip=list(c.shape),frame=list(f.shape),query=list(q.shape),all_masks_valid=True)
    try:
        with torch.no_grad(),Record():
            frames,clips=model.encode_context(c,f,fm)
            args=[q,qm,None,clips,frames]
            if case['method'] in ('MS-SL','MS-SL++','SDM','Holmes-multi'): args.append(fm)
            scores=model.get_pred_from_raw_query(*args)
        case['inference_status']='PASS'
        case['encoded_shapes']=dict(clip=list(clips.shape),frame=list(frames.shape))
        case['synthetic_score_shapes']=[list(s.shape) for s in scores if isinstance(s,torch.Tensor)]
    except Exception:
        case['inference_status']='FAILED'
        case['inference_error']=traceback.format_exc()
    finally:
        for h in handles: h.remove()
    own_called={id(p) for n,m in model.named_modules() if n in called for p in m.parameters(recurse=False)}
    case['hook_called_modules']=called
    case['hook_parameter_elements']=sum(p.numel() for p in params.values() if id(p) in own_called)
    case['functional_parameter_names_beyond_hooks']=[n for n,p in params.items() if id(p) in used-own_called]
    case['called_but_not_tensor_accessed_names']=[n for n,p in params.items() if id(p) in own_called-used]
    case['parameters']=dict(training_total=sum(p.numel() for p in params.values()),
        trainable=sum(p.numel() for p in params.values() if p.requires_grad),
        inference=sum(p.numel() for p in params.values() if id(p) in used) if case['inference_status']=='PASS' else None,
        buffer_elements=sum(b.numel() for b in model.buffers()),
        unique_parameter_tensors=len(params),registered_parameter_aliases=sum(map(len,aliases.values())))
    case['parameter_manifest']=[]
    for n,p in params.items():
        case['parameter_manifest'].append(dict(name=n,aliases=aliases[id(p)],shape=list(p.shape),numel=p.numel(),
            requires_grad=p.requires_grad,group=group(n),inference_accessed=id(p) in used,operations=sorted(used_ops.get(n,[]))))
    case['breakdown']={}
    for g in ('query','clip','frame','fusion','other'):
        ps=[x for x in case['parameter_manifest'] if x['group']==g]
        case['breakdown'][g]=dict(training_total=sum(x['numel'] for x in ps),
            trainable=sum(x['numel'] for x in ps if x['requires_grad']),
            inference=sum(x['numel'] for x in ps if x['inference_accessed']) if case['inference_status']=='PASS' else None)
    case['module_parameter_sources']=module_ref
    case['unused_inference_parameters']=[x['name'] for x in case['parameter_manifest'] if not x['inference_accessed']]
    case['unused_inference_groups']={}
    for x in case['parameter_manifest']:
        if not x['inference_accessed']:
            key=x['name'].split('.')[0]
            case['unused_inference_groups'][key]=case['unused_inference_groups'].get(key,0)+x['numel']
    return params
