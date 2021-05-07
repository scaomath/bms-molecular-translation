#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nfnet.py

from common import *
import math
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Tuple, Optional
from functools import partial

from itertools import repeat
import collections.abc



#--------------------------------------------------------------------------


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class FastAdaptiveAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(self.flatten)
            self.flatten = False
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return self.pool_type == ''

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'


def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, pool_type='avg', use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting issue
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0., use_conv=False):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten_after_fc = use_conv and pool_type

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x
#--------------------------------------------------------------------------

def gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return F.gelu(x)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()

def is_no_jit() : return False
def is_scriptable() : return False
def is_exportable() : return False
_ACT_FN_ME = dict(
)
_ACT_FN_JIT = dict(
)
_ACT_FN_DEFAULT = dict(
    gelu=gelu,
)

_ACT_LAYER_ME = dict(
)
_ACT_LAYER_JIT = dict(
)
_ACT_LAYER_DEFAULT = dict(
    gelu=GELU,
    sigmoid=Sigmoid,
)



def get_act_fn(name='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        # If not exporting or scripting the model, first look for a memory-efficient version with
        # custom autograd, then fallback
        if name in _ACT_FN_ME:
            return _ACT_FN_ME[name]
    if is_exportable() and name in ('silu', 'swish'):
        # FIXME PyTorch SiLU doesn't ONNX export, this is a temp hack
        return swish
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_FN_JIT:
            return _ACT_FN_JIT[name]
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    if is_exportable() and name in ('silu', 'swish'):
        # FIXME PyTorch SiLU doesn't ONNX export, this is a temp hack
        return Swish
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name, inplace=False, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is not None:
        return act_layer(inplace=inplace, **kwargs)
    else:
        return None
#--------------------------------------------------------------------------

# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic
#--------------------------------------------------------------------------
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#--------------------------------------------------------------------------
class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * min_channels can be specified to keep reduced channel count at a minimum (default: 8)
        * divisor can be specified to keep channels rounded to specified values (default: 1)
        * reduction channels can be specified directly by arg (if reduction_channels is set)
        * reduction channels can be specified by float ratio (if reduction_ratio is set)
    """
    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, gate_layer='sigmoid',
                 reduction_ratio=None, reduction_channels=None, min_channels=8, divisor=1):
        super(SEModule, self).__init__()
        if reduction_channels is not None:
            reduction_channels = reduction_channels  # direct specification highest priority, no rounding/min done
        elif reduction_ratio is not None:
            reduction_channels = make_divisible(channels * reduction_ratio, divisor, min_channels)
        else:
            reduction_channels = make_divisible(channels // reduction, divisor, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

#--------------------------------------------------------------------------

def get_attn(attn_type):
    if isinstance(attn_type, torch.nn.Module):
        return attn_type
    module_cls = None
    if attn_type is not None:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule
            else:
                assert False, "Invalid attn module (%s)" % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls




#--------------------------------------------------------------------------
class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.
    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding=None, dilation=1,
            groups=1, bias=False, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return weight

    def forward(self, x):
        x = F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.
    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding='SAME', dilation=1,
            groups=1, bias=False, eps=1e-5):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.same_pad = is_dynamic
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return weight

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        x = F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692
    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=True, gamma=1.0, eps=1e-5, gain_init=1.0, use_layernorm=False):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm  # experimental, slightly faster/less GPU memory to hijack LN kernel

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692
    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding='SAME', dilation=1, groups=1,
            bias=True, gamma=1.0, eps=1e-5, gain_init=1.0, use_layernorm=False):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm  # experimental, slightly faster/less GPU memory to hijack LN kernel

    # NOTE an alternate formulation to consider, closer to DeepMind Haiku impl but doesn't seem
    # to make much numerical difference (+/- .002 to .004) in top-1 during eval.
    # def get_weight(self):
    #         var, mean = torch.var_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
    #         scale = torch.rsqrt((self.weight[0].numel() * var).clamp_(self.eps)) * self.gain
    #         weight = (self.weight - mean) * scale
    #     return self.gain * weight

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)








#--------------------------------------------------------------------------

@dataclass
class NfCfg:
    depths: Tuple[int, int, int, int]
    channels: Tuple[int, int, int, int]
    alpha: float = 0.2
    stem_type: str = '3x3'
    stem_chs: Optional[int] = None
    group_size: Optional[int] = None
    attn_layer: Optional[str] = None
    attn_kwargs: dict = None
    attn_gain: float = 2.0  # NF correction gain to apply if attn layer is used
    width_factor: float = 1.0
    bottle_ratio: float = 0.5
    num_features: int = 0  # num out_channels for final conv, no final_conv if 0
    ch_div: int = 8  # round channels % 8 == 0 to keep tensor-core use optimal
    reg: bool = False  # enables EfficientNet-like options used in RegNet variants, expand from in_chs, se in middle
    extra_conv: bool = False  # extra 3x3 bottleneck convolution for NFNet models
    gamma_in_act: bool = False
    same_padding: bool = False
    skipinit: bool = False  # disabled by default, non-trivial performance impact
    zero_init_fc: bool = False
    act_layer: str = 'silu'


def _nfres_cfg(
        depths, channels=(256, 512, 1024, 2048), group_size=None, act_layer='relu', attn_layer=None, attn_kwargs=None):
    attn_kwargs = attn_kwargs or {}
    cfg = NfCfg(
        depths=depths, channels=channels, stem_type='7x7_pool', stem_chs=64, bottle_ratio=0.25,
        group_size=group_size, act_layer=act_layer, attn_layer=attn_layer, attn_kwargs=attn_kwargs)
    return cfg


def _nfreg_cfg(depths, channels=(48, 104, 208, 440)):
    num_features = 1280 * channels[-1] // 440
    attn_kwargs = dict(reduction_ratio=0.5, divisor=8)
    cfg = NfCfg(
        depths=depths, channels=channels, stem_type='3x3', group_size=8, width_factor=0.75, bottle_ratio=2.25,
        num_features=num_features, reg=True, attn_layer='se', attn_kwargs=attn_kwargs)
    return cfg


def _nfnet_cfg(
        depths, channels=(256, 512, 1536, 1536), group_size=128, bottle_ratio=0.5, feat_mult=2.,
        act_layer='gelu', attn_layer='se', attn_kwargs=None):
    num_features = int(channels[-1] * feat_mult)
    attn_kwargs = attn_kwargs if attn_kwargs is not None else dict(reduction_ratio=0.5, divisor=8)
    cfg = NfCfg(
        depths=depths, channels=channels, stem_type='deep_quad', stem_chs=128, group_size=group_size,
        bottle_ratio=bottle_ratio, extra_conv=True, num_features=num_features, act_layer=act_layer,
        attn_layer=attn_layer, attn_kwargs=attn_kwargs)
    return cfg


def _dm_nfnet_cfg(depths, channels=(256, 512, 1536, 1536), act_layer='gelu', skipinit=True):
    attn_kwargs = dict(reduction_ratio=0.5, divisor=8)
    cfg = NfCfg(
        depths=depths, channels=channels, stem_type='deep_quad', stem_chs=128, group_size=128,
        bottle_ratio=0.5, extra_conv=True, gamma_in_act=True, same_padding=True, skipinit=skipinit,
        num_features=int(channels[-1] * 2.0), act_layer=act_layer, attn_layer='se', attn_kwargs=attn_kwargs)
    return cfg


model_cfgs = dict(
    # NFNet-F models w/ GELU compatible with DeepMind weights
    dm_nfnet_f0=_dm_nfnet_cfg(depths=(1, 2, 6, 3)),
    dm_nfnet_f1=_dm_nfnet_cfg(depths=(2, 4, 12, 6)),
    dm_nfnet_f2=_dm_nfnet_cfg(depths=(3, 6, 18, 9)),
    dm_nfnet_f3=_dm_nfnet_cfg(depths=(4, 8, 24, 12)),
    dm_nfnet_f4=_dm_nfnet_cfg(depths=(5, 10, 30, 15)),
    dm_nfnet_f5=_dm_nfnet_cfg(depths=(6, 12, 36, 18)),
    dm_nfnet_f6=_dm_nfnet_cfg(depths=(7, 14, 42, 21)),

    # NFNet-F models w/ GELU (I will likely deprecate/remove these models and just keep dm_ ver for GELU)
    nfnet_f0=_nfnet_cfg(depths=(1, 2, 6, 3)),
    nfnet_f1=_nfnet_cfg(depths=(2, 4, 12, 6)),
    nfnet_f2=_nfnet_cfg(depths=(3, 6, 18, 9)),
    nfnet_f3=_nfnet_cfg(depths=(4, 8, 24, 12)),
    nfnet_f4=_nfnet_cfg(depths=(5, 10, 30, 15)),
    nfnet_f5=_nfnet_cfg(depths=(6, 12, 36, 18)),
    nfnet_f6=_nfnet_cfg(depths=(7, 14, 42, 21)),
    nfnet_f7=_nfnet_cfg(depths=(8, 16, 48, 24)),

    # NFNet-F models w/ SiLU (much faster in PyTorch)
    nfnet_f0s=_nfnet_cfg(depths=(1, 2, 6, 3), act_layer='silu'),
    nfnet_f1s=_nfnet_cfg(depths=(2, 4, 12, 6), act_layer='silu'),
    nfnet_f2s=_nfnet_cfg(depths=(3, 6, 18, 9), act_layer='silu'),
    nfnet_f3s=_nfnet_cfg(depths=(4, 8, 24, 12), act_layer='silu'),
    nfnet_f4s=_nfnet_cfg(depths=(5, 10, 30, 15), act_layer='silu'),
    nfnet_f5s=_nfnet_cfg(depths=(6, 12, 36, 18), act_layer='silu'),
    nfnet_f6s=_nfnet_cfg(depths=(7, 14, 42, 21), act_layer='silu'),
    nfnet_f7s=_nfnet_cfg(depths=(8, 16, 48, 24), act_layer='silu'),

    # Experimental 'light' versions of nfnet-f that are little leaner
    nfnet_l0a=_nfnet_cfg(
        depths=(1, 2, 6, 3), channels=(256, 512, 1280, 1536), feat_mult=1.5, group_size=64, bottle_ratio=0.25,
        attn_kwargs=dict(reduction_ratio=0.25, divisor=8), act_layer='silu'),
    nfnet_l0b=_nfnet_cfg(
        depths=(1, 2, 6, 3), channels=(256, 512, 1536, 1536), feat_mult=1.5, group_size=64, bottle_ratio=0.25,
        attn_kwargs=dict(reduction_ratio=0.25, divisor=8), act_layer='silu'),
    eca_nfnet_l0=_nfnet_cfg(
        depths=(1, 2, 6, 3), channels=(256, 512, 1536, 1536), feat_mult=1.5, group_size=64, bottle_ratio=0.25,
        attn_layer='eca', attn_kwargs=dict(), act_layer='silu'),

    # EffNet influenced RegNet defs.
    # NOTE: These aren't quite the official ver, ch_div=1 must be set for exact ch counts. I round to ch_div=8.
    nf_regnet_b0=_nfreg_cfg(depths=(1, 3, 6, 6)),
    nf_regnet_b1=_nfreg_cfg(depths=(2, 4, 7, 7)),
    nf_regnet_b2=_nfreg_cfg(depths=(2, 4, 8, 8), channels=(56, 112, 232, 488)),
    nf_regnet_b3=_nfreg_cfg(depths=(2, 5, 9, 9), channels=(56, 128, 248, 528)),
    nf_regnet_b4=_nfreg_cfg(depths=(2, 6, 11, 11), channels=(64, 144, 288, 616)),
    nf_regnet_b5=_nfreg_cfg(depths=(3, 7, 14, 14), channels=(80, 168, 336, 704)),
    # FIXME add B6-B8

    # ResNet (preact, D style deep stem/avg down) defs
    nf_resnet26=_nfres_cfg(depths=(2, 2, 2, 2)),
    nf_resnet50=_nfres_cfg(depths=(3, 4, 6, 3)),
    nf_resnet101=_nfres_cfg(depths=(3, 4, 23, 3)),

    nf_seresnet26=_nfres_cfg(depths=(2, 2, 2, 2), attn_layer='se', attn_kwargs=dict(reduction_ratio=1/16)),
    nf_seresnet50=_nfres_cfg(depths=(3, 4, 6, 3), attn_layer='se', attn_kwargs=dict(reduction_ratio=1/16)),
    nf_seresnet101=_nfres_cfg(depths=(3, 4, 23, 3), attn_layer='se', attn_kwargs=dict(reduction_ratio=1/16)),

    nf_ecaresnet26=_nfres_cfg(depths=(2, 2, 2, 2), attn_layer='eca', attn_kwargs=dict()),
    nf_ecaresnet50=_nfres_cfg(depths=(3, 4, 6, 3), attn_layer='eca', attn_kwargs=dict()),
    nf_ecaresnet101=_nfres_cfg(depths=(3, 4, 23, 3), attn_layer='eca', attn_kwargs=dict()),

)


class GammaAct(nn.Module):
    def __init__(self, act_type='relu', gamma: float = 1.0, inplace=False):
        super().__init__()
        self.act_fn = get_act_fn(act_type)
        self.gamma = gamma
        self.inplace = inplace

    def forward(self, x):
        return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)


def act_with_gamma(act_type, gamma: float = 1.):
    def _create(inplace=False):
        return GammaAct(act_type, gamma=gamma, inplace=inplace)
    return _create


class DownsampleAvg(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, conv_layer=ScaledStdConv2d):
        """ AvgPool Downsampling as in 'D' ResNet variants. Support for dilation."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)

    def forward(self, x):
        return self.conv(self.pool(x))


class NormFreeBlock(nn.Module):
    """Normalization-Free pre-activation block.
    """

    def __init__(
            self, in_chs, out_chs=None, stride=1, dilation=1, first_dilation=None,
            alpha=1.0, beta=1.0, bottle_ratio=0.25, group_size=None, ch_div=1, reg=True, extra_conv=False,
            skipinit=False, attn_layer=None, attn_gain=2.0, act_layer=None, conv_layer=None, drop_path_rate=0.):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_chs = out_chs or in_chs
        # RegNet variants scale bottleneck from in_chs, otherwise scale from out_chs like ResNet
        mid_chs = make_divisible(in_chs * bottle_ratio if reg else out_chs * bottle_ratio, ch_div)
        groups = 1 if not group_size else mid_chs // group_size
        if group_size and group_size % ch_div == 0:
            mid_chs = group_size * groups  # correct mid_chs if group_size divisible by ch_div, otherwise error
        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain

        if in_chs != out_chs or stride != 1 or dilation != first_dilation:
            self.downsample = DownsampleAvg(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, conv_layer=conv_layer)
        else:
            self.downsample = None

        self.act1 = act_layer()
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.act2 = act_layer(inplace=True)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        if extra_conv:
            self.act2b = act_layer(inplace=True)
            self.conv2b = conv_layer(mid_chs, mid_chs, 3, stride=1, dilation=dilation, groups=groups)
        else:
            self.act2b = None
            self.conv2b = None
        if reg and attn_layer is not None:
            self.attn = attn_layer(mid_chs)  # RegNet blocks apply attn btw conv2 & 3
        else:
            self.attn = None
        self.act3 = act_layer()
        self.conv3 = conv_layer(mid_chs, out_chs, 1, gain_init=1. if skipinit else 0.)
        if not reg and attn_layer is not None:
            self.attn_last = attn_layer(out_chs)  # ResNet blocks apply attn after conv3
        else:
            self.attn_last = None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.skipinit_gain = nn.Parameter(torch.tensor(0.)) if skipinit else None

    def forward(self, x):
        out = self.act1(x) * self.beta

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)

        # residual branch
        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        if self.conv2b is not None:
            out = self.conv2b(self.act2b(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act3(out))
        if self.attn_last is not None:
            out = self.attn_gain * self.attn_last(out)
        out = self.drop_path(out)

        if self.skipinit_gain is not None:
            out.mul_(self.skipinit_gain)  # this slows things down more than expected, TBD
        out = out * self.alpha + shortcut
        return out


def create_stem(in_chs, out_chs, stem_type='', conv_layer=None, act_layer=None, preact_feature=True):
    stem_stride = 2
    stem_feature = dict(num_chs=out_chs, reduction=2, module='stem.conv')
    stem = OrderedDict()
    assert stem_type in ('', 'deep', 'deep_tiered', 'deep_quad', '3x3', '7x7', 'deep_pool', '3x3_pool', '7x7_pool')
    if 'deep' in stem_type:
        if 'quad' in stem_type:
            # 4 deep conv stack as in NFNet-F models
            assert not 'pool' in stem_type
            stem_chs = (out_chs // 8, out_chs // 4, out_chs // 2, out_chs)
            strides = (2, 1, 1, 2)
            stem_stride = 4
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv3')
        else:
            if 'tiered' in stem_type:
                stem_chs = (3 * out_chs // 8, out_chs // 2, out_chs)  # 'T' resnets in resnet.py
            else:
                stem_chs = (out_chs // 2, out_chs // 2, out_chs)  # 'D' ResNets
            strides = (2, 1, 1)
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv2')
        last_idx = len(stem_chs) - 1
        for i, (c, s) in enumerate(zip(stem_chs, strides)):
            stem[f'conv{i + 1}'] = conv_layer(in_chs, c, kernel_size=3, stride=s)
            if i != last_idx:
                stem[f'act{i + 2}'] = act_layer(inplace=True)
            in_chs = c
    elif '3x3' in stem_type:
        # 3x3 stem conv as in RegNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=3, stride=2)
    else:
        # 7x7 stem conv as in ResNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)

    if 'pool' in stem_type:
        stem['pool'] = nn.MaxPool2d(3, stride=2, padding=1)
        stem_stride = 4

    return nn.Sequential(stem), stem_stride, stem_feature


# from https://github.com/deepmind/deepmind-research/tree/master/nfnets
_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)


class NormFreeNet(nn.Module):
    """ Normalization-Free Network
    As described in :
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    and
    `High-Performance Large-Scale Image Recognition Without Normalization` - https://arxiv.org/abs/2102.06171
    This model aims to cover both the NFRegNet-Bx models as detailed in the paper's code snippets and
    the (preact) ResNet models described earlier in the paper.
    There are a few differences:
        * channels are rounded to be divisible by 8 by default (keep tensor core kernels happy),
            this changes channel dim and param counts slightly from the paper models
        * activation correcting gamma constants are moved into the ScaledStdConv as it has less performance
            impact in PyTorch when done with the weight scaling there. This likely wasn't a concern in the JAX impl.
        * a config option `gamma_in_act` can be enabled to not apply gamma in StdConv as described above, but
            apply it in each activation. This is slightly slower, numerically different, but matches official impl.
        * skipinit is disabled by default, it seems to have a rather drastic impact on GPU memory use and throughput
            for what it is/does. Approx 8-10% throughput loss.
    """
    def __init__(self, cfg: NfCfg, num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
                 drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert cfg.act_layer in _nonlin_gamma, f"Please add non-linearity constants for activation ({cfg.act_layer})."
        conv_layer = ScaledStdConv2dSame if cfg.same_padding else ScaledStdConv2d
        if cfg.gamma_in_act:
            act_layer = act_with_gamma(cfg.act_layer, gamma=_nonlin_gamma[cfg.act_layer])
            conv_layer = partial(conv_layer, eps=1e-4)  # DM weights better with higher eps
        else:
            act_layer = get_act_layer(cfg.act_layer)
            conv_layer = partial(conv_layer, gamma=_nonlin_gamma[cfg.act_layer])
        attn_layer = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None

        stem_chs = make_divisible((cfg.stem_chs or cfg.channels[0]) * cfg.width_factor, cfg.ch_div)
        self.stem, stem_stride, stem_feat = create_stem(
            in_chans, stem_chs, cfg.stem_type, conv_layer=conv_layer, act_layer=act_layer)

        self.feature_info = [stem_feat]
        drop_path_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        prev_chs = stem_chs
        net_stride = stem_stride
        dilation = 1
        expected_var = 1.0
        stages = []
        for stage_idx, stage_depth in enumerate(cfg.depths):
            stride = 1 if stage_idx == 0 and stem_stride > 2 else 2
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2

            blocks = []
            for block_idx in range(cfg.depths[stage_idx]):
                first_block = block_idx == 0 and stage_idx == 0
                out_chs = make_divisible(cfg.channels[stage_idx] * cfg.width_factor, cfg.ch_div)
                blocks += [NormFreeBlock(
                    in_chs=prev_chs, out_chs=out_chs,
                    alpha=cfg.alpha,
                    beta=1. / expected_var ** 0.5,
                    stride=stride if block_idx == 0 else 1,
                    dilation=dilation,
                    first_dilation=first_dilation,
                    group_size=cfg.group_size,
                    bottle_ratio=1. if cfg.reg and first_block else cfg.bottle_ratio,
                    ch_div=cfg.ch_div,
                    reg=cfg.reg,
                    extra_conv=cfg.extra_conv,
                    skipinit=cfg.skipinit,
                    attn_layer=attn_layer,
                    attn_gain=cfg.attn_gain,
                    act_layer=act_layer,
                    conv_layer=conv_layer,
                    drop_path_rate=drop_path_rates[stage_idx][block_idx],
                )]
                if block_idx == 0:
                    expected_var = 1.  # expected var is reset after first block of each stage
                expected_var += cfg.alpha ** 2   # Even if reset occurs, increment expected variance
                first_dilation = dilation
                prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')]
            stages += [nn.Sequential(*blocks)]
        self.stages = nn.Sequential(*stages)

        if cfg.num_features:
            # The paper NFRegNet models have an EfficientNet-like final head convolution.
            self.num_features = make_divisible(cfg.width_factor * cfg.num_features, cfg.ch_div)
            self.final_conv = conv_layer(prev_chs, self.num_features, 1)
            self.feature_info[-1] = dict(num_chs=self.num_features, reduction=net_stride, module=f'final_conv')
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.final_act = act_layer(inplace=cfg.num_features > 0)

        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

        for n, m in self.named_modules():
            if 'fc' in n and isinstance(m, nn.Linear):
                if cfg.zero_init_fc:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# def _create_normfreenet(variant, pretrained=False, **kwargs):
#     model_cfg = model_cfgs[variant]
#     feature_cfg = dict(flatten_sequential=True)
#     return build_model_with_cfg(
#         NormFreeNet, variant, pretrained,
#         default_cfg=default_cfgs[variant],
#         model_cfg=model_cfg,
#         feature_cfg=feature_cfg,
#         **kwargs)



#############################################################################
# NfCfg(depths=(1, 2, 6, 3), channels=(256, 512, 1536, 1536), alpha=0.2, stem_type='deep_quad',
# stem_chs=128, group_size=128, attn_layer='se', attn_kwargs={'reduction_ratio': 0.5, 'divisor': 8},
# attn_gain=2.0, width_factor=1.0, bottle_ratio=0.5, num_features=3072, ch_div=8,
# reg=False, extra_conv=True, gamma_in_act=True, same_padding=True, skipinit=True,
# zero_init_fc=False, act_layer='gelu')

def make_nfnet_f0():
    #net = _create_normfreenet('dm_nfnet_f0', pretrained=False)
    net = NormFreeNet(
        model_cfgs['dm_nfnet_f0'],
        num_classes = 1000,
        in_chans = 3,
        global_pool = 'avg',
        output_stride = 32,
        drop_rate = 0.,
        drop_path_rate = 0.
    )
    return net

PRETRAIN_CHECKPOINT = '../pretrain_model/dm_nfnet_f0-604f9c3a.pth'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)
IMAGE_RGB_MEAN = IMAGENET_DEFAULT_MEAN
IMAGE_RGB_STD  = IMAGENET_DEFAULT_STD




#





def run_check_pretrain_net():
    net = make_nfnet_f0()
    print(net)
    pretrain_state_dict = torch.load(PRETRAIN_CHECKPOINT, map_location=lambda storage, loc: storage)
    #state_dict = net.state_dict()

    s = net.load_state_dict(pretrain_state_dict, strict=True)
    print(s)


    #---
    if 1:
        net = net.cuda().eval()

        synset_file = '/root/share/data/imagenet/dummy/synset_words'
        synset = read_list_from_file(synset_file)
        synset = [s[10:].split(',')[0] for s in synset]

        image_dir ='/root/share/data/imagenet/dummy/256x256'
        for f in [
            'great_white_shark','screwdriver','ostrich','blad_eagle','english_foxhound','goldfish',
        ]:
            image_file = image_dir +'/%s.jpg'%f
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            #image = cv2.resize(image,dsize=(320,320),interpolation=cv2.INTER_CUBIC)
            image = image[16:16+224,16:16+224]


            image = image[:,:,::-1]
            image = image.astype(np.float32)/255
            image = (image -IMAGE_RGB_MEAN)/IMAGE_RGB_STD
            input = image.transpose(2,0,1)
            input = torch.from_numpy(input).float().cuda().unsqueeze(0)

            logit = net(input)
            proability = F.softmax(logit,-1)

            probability = proability.data.cpu().numpy().reshape(-1)
            argsort = np.argsort(-probability)

            print(f, image.shape)
            print(probability[:5])
            for t in range(5):
                print(t, '%24s'%synset[argsort[t]][:24], '%3d'%argsort[t], probability[argsort[t]])
            print('')



if __name__ == '__main__':
    run_check_pretrain_net()