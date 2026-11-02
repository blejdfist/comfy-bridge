"""Benchmark-only timing hooks; no ComfyUI node definitions or registration.

Runs in place of ComfyUI's main.py; must not import comfyui_modal.
"""
import builtins
import contextvars
import functools
import json
import runpy
import sys
import time
from pathlib import Path

RECORDS = Path('/tmp/benchmark-records.jsonl')
TARGETS = [
    ('comfy.utils', None, 'load_torch_file', 'model_file_read'),
    ('comfy.model_management', None, 'load_models_gpu', 'weight_loading'),
    ('comfy.samplers', 'KSAMPLER', 'sample', 'sampling'),
    ('comfy.sd', 'VAE', 'decode', 'vae_decode'),
    ('comfy.sd', 'VAE', 'encode', 'vae_encode'),
    ('execution', None, 'execute', 'node_execution'),
]
original_import = builtins.__import__
patched = set()
active_node = contextvars.ContextVar('benchmark_active_node', default=None)


def record(row):
    with RECORDS.open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def measured_node(original):
    """Time engine execution attempts; tag nested hooks without recording inputs.

    PENDING attempts remain separate records for lazy/async nodes. A synchronous
    successful node has one inclusive duration; nested stage durations overlap it.
    """
    @functools.wraps(original)
    async def call(server, dynprompt, caches, current_item, *args, **kwargs):
        import torch
        context = {'node_id': str(current_item),
                   'class_type': dynprompt.get_node(current_item)['class_type']}
        token = active_node.set(context)
        result = None
        cuda = torch.cuda.is_available()
        try:
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                result = await original(server, dynprompt, caches, current_item, *args, **kwargs)
                return result
            finally:
                if cuda:
                    torch.cuda.synchronize()
                end = time.perf_counter()
                state = getattr(result[0], 'name', str(result[0])) if result else 'EXCEPTION'
                record({**context, 'kind': 'node_execution', 'start': start, 'end': end,
                        'seconds': end - start, 'execution_result': state})
        finally:
            active_node.reset(token)
    return call


def measured(original, kind):
    @functools.wraps(original)
    def call(*args, **kwargs):
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            if kind == 'sampling':
                torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if kind == 'sampling':
            previous = start
            callback = args[4] if len(args) > 4 else kwargs.get('callback')

            def step_callback(step, denoised, latent, total_steps):
                nonlocal previous
                if cuda:
                    torch.cuda.synchronize()
                now = time.perf_counter()
                record({'kind': 'sampling_step', 'step': step, 'total_steps': total_steps,
                        'start': previous, 'end': now, 'seconds': now - previous})
                previous = now
                if callback is not None:
                    return callback(step, denoised, latent, total_steps)
            if len(args) > 4:
                args = (*args[:4], step_callback, *args[5:])
            else:
                kwargs['callback'] = step_callback
        try:
            return original(*args, **kwargs)
        finally:
            if cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            extra = {}
            if kind == 'sampling' and cuda:
                extra = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            record({**(active_node.get() or {}), 'kind': kind, 'start': start, 'end': end,
                    'seconds': end - start, **extra})
    return call


def install():
    if 'torch' not in sys.modules:
        return
    for module_name, owner_name, name, kind in TARGETS:
        module = sys.modules.get(module_name)
        owner = getattr(module, owner_name, None) if owner_name else module
        if kind in patched or owner is None or not hasattr(owner, name):
            continue
        wrapper = measured_node(getattr(owner, name)) if kind == 'node_execution' else measured(getattr(owner, name), kind)
        setattr(owner, name, wrapper)
        patched.add(kind)
    if len(patched) == len(TARGETS):
        builtins.__import__ = original_import


def hooked_import(*args, **kwargs):
    result = original_import(*args, **kwargs)
    install()
    return result


if __name__ == '__main__':
    sys.path.insert(0, '/opt/ComfyUI')
    builtins.__import__ = hooked_import
    sys.argv[0] = '/opt/ComfyUI/main.py'
    runpy.run_path(sys.argv[0], run_name='__main__')
