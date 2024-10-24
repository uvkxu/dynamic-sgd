from __future__ import annotations

from copy import deepcopy
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module

def exists(val):
    return val is not None

def divisible_by(num, den):
    return (num % den) == 0

def get_module_device(m: Module):
    return next(m.parameters()).device

def maybe_coerce_dtype(t, dtype):
    if t.dtype == dtype:
        return t
    return t.to(dtype)

def inplace_copy(tgt: Tensor, src: Tensor, *, auto_move_device=False, coerce_dtype=False):
    if auto_move_device:
        src = src.to(tgt.device)
    if coerce_dtype:
        src = maybe_coerce_dtype(src, tgt.dtype)
    tgt.copy_(src)

def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, auto_move_device=False, coerce_dtype=False):
    if auto_move_device:
        src = src.to(tgt.device)
    if coerce_dtype:
        src = maybe_coerce_dtype(src, tgt.dtype)
    tgt.lerp_(src, weight)

class EMA(Module):
    def __init__(
        self,
        model: Module,
        ema_model: Module | None = None,
        beta=0.9999,
        update_after_step=100,
        update_every=10,
        inv_gamma=1.0,
        power=2/3,
        min_value=0.0,
        param_or_buffer_names_no_ema: set[str] = set(),
        ignore_names: set[str] = set(),
        ignore_startswith_names: set[str] = set(),
        include_online_model=True,
        allow_different_devices=False,
        use_foreach=False,
        update_model_with_ema_every=None,
        update_model_with_ema_beta=0.,
        forward_method_names: tuple[str, ...] = (),
        move_ema_to_online_device=False,
        coerce_dtype=False,
        lazy_init_ema=False
    ):
        super().__init__()
        self.beta = beta
        self.is_frozen = beta == 1.0
        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]  # hack

        self.ema_model = None
        self.forward_method_names = forward_method_names

        if not lazy_init_ema:
            self.init_ema(ema_model)
        else:
            assert not exists(ema_model)

        self.inplace_copy = partial(inplace_copy, auto_move_device=allow_different_devices, coerce_dtype=coerce_dtype)
        self.inplace_lerp = partial(inplace_lerp, auto_move_device=allow_different_devices, coerce_dtype=coerce_dtype)

        self.update_every = update_every
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema
        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        self.update_model_with_ema_every = update_model_with_ema_every
        self.update_model_with_ema_beta = update_model_with_ema_beta

        self.allow_different_devices = allow_different_devices
        self.coerce_dtype = coerce_dtype
        self.move_ema_to_online_device = move_ema_to_online_device

        self.use_foreach = use_foreach

        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))

    def init_ema(self, ema_model: Module | None = None):
        self.ema_model = ema_model
        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(self.model)
            except Exception as e:
                print(f"Error: While trying to deepcopy model: {e}")
                exit()

        for p in self.ema_model.parameters():
            p.detach_()

        for forward_method_name in self.forward_method_names:
            fn = getattr(self.ema_model, forward_method_name)
            setattr(self, forward_method_name, fn)

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}

    def add_to_optimizer_post_step_hook(self, optimizer):
        assert hasattr(optimizer, 'register_step_post_hook')
        def hook(*_):
            self.update()
        return optimizer.register_step_post_hook(hook)

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def eval(self):
        return self.ema_model.eval()

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        training = self.ema_model.training
        out = self.ema_model(*args, **kwargs)
        self.ema_model.train(training)
        return out

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(ma_params.data, current_params.data)
        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(ma_buffers.data, current_buffers.data)

    def copy_params_from_ema_to_model(self):
        copy = self.inplace_copy
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(current_params.data, ma_params.data)
        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(current_buffers.data, ma_buffers.data)

    def update_model_with_ema(self, decay=None):
        if not exists(decay):
            decay = self.update_model_with_ema_beta
        if decay == 0.0:
            return self.copy_params_from_ema_to_model()
        self.update_moving_average(self.model, self.ema_model, decay)

    def get_current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        if epoch.item() <= 0:
            return 0.0
        return value.clamp(min=self.min_value, max=self.beta).item()

    def update(self):
        step = self.step.item()
        self.step += 1

        if not self.initted.item():
            if not exists(self.ema_model):
                self.init_ema()
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))
            return

        should_update = divisible_by(step, self.update_every)

        if should_update and step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if should_update:
            self.update_moving_average(self.ema_model, self.model)

        if exists(self.update_model_with_ema_every) and divisible_by(step, self.update_model_with_ema_every):
            self.update_model_with_ema()

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model, current_decay=None):
        if self.is_frozen:
            return

        if self.move_ema_to_online_device and get_module_device(ma_model) != get_module_device(current_model):
            ma_model.to(get_module_device(current_model))

        if not exists(current_decay):
            current_decay = self.get_current_decay()

        tensors_to_copy = []
        tensors_to_lerp = []

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.param_or_buffer_names_no_ema or name in self.ignore_names or any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue
            if not self.use_foreach:
                self.inplace_lerp(ma_params.data, current_params.data, 1 - current_decay)
            else:
                tensors_to_copy.append(current_params.data)
                tensors_to_lerp.append(ma_params.data)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.param_or_buffer_names_no_ema or name in self.ignore_names or any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue
            if not self.use_foreach:
                self.inplace_lerp(ma_buffer.data, current_buffer.data, 1 - current_decay)
            else:
                tensors_to_copy.append(current_buffer.data)
                tensors_to_lerp.append(ma_buffer.data)

        if len(tensors_to_copy) > 0:
            if self.use_foreach:
                torch._foreach_lerp_(tensors_to_lerp, tensors_to_copy, 1 - current_decay)
