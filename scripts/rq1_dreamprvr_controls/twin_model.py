"""Numerical implementation used by the reported experiment."""
from Models.DreamPRVR.model import DreamPRVR_Net
from easydict import EasyDict
import ast
import inspect
from torch import nn
import textwrap


def model_config(cfg):
    return EasyDict(visual_input_size=cfg['visual_feat_dim'], query_input_size=cfg['q_feat_size'],
                   **{key: cfg[key] for key in (
                       'hidden_size', 'max_ctx_l', 'max_desc_l', 'map_size', 'input_drop', 'drop',
                       'n_heads', 'initializer_range', 'margin', 'hard_pool_size', 'sft_factor',
                       'num_registers', 'timesteps', 'attention_num', 'num_diffusion_layers')},
                   use_hard_negative=False)


def twin_context_function():
    """Preserve every upstream sampling/control-flow operation, replace only L."""
    original = DreamPRVR_Net.encode_context
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    count = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and node.value == 128:
            node.value = 32
            count += 1
    assert count == 2, 'Upstream padding block changed'
    prefix = ast.parse('''
frame_video_feat = clip_video_feat
assert frame_video_feat.shape[1] == 32
video_mask = torch.ones(frame_video_feat.shape[:2], device=frame_video_feat.device, dtype=frame_video_feat.dtype)
''').body
    function.body = prefix+function.body
    scope = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), scope)
    return scope['encode_context']


class TwoClipDream(DreamPRVR_Net):
    encode_context = twin_context_function()


def build(cfg):
    # Always construct the complete original model first. Slicing consumes no RNG.
    model = DreamPRVR_Net(model_config(cfg))
    if cfg['variant'] == 'twin':
        model.__class__ = TwoClipDream
        branch = model.frame_encoder
        branch.clip_num = 32
        branch.layer2.weight = nn.Parameter(branch.layer2.weight[:32].detach().clone())
        branch.layer2.bias = nn.Parameter(branch.layer2.bias[:32].detach().clone())
        branch.layer2.out_features = 32
        for module in branch.modules():
            if hasattr(module, 'frame_len'):
                module.frame_len = 32
            if hasattr(module, 'config') and 'frame_len' in module.config:
                module.config.frame_len = 32
    assert isinstance(model, DreamPRVR_Net)
    return model
