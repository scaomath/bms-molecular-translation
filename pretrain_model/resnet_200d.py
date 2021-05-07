from common import *
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGE_RGB_MEAN = IMAGENET_DEFAULT_MEAN
IMAGE_RGB_STD  = IMAGENET_DEFAULT_STD

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs={
    'resnet200d_320': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 320, 320), crop_pct=1.0, pool_size=(10, 10)),
}

#########################################################################################

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


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1

def create_attn(attn_type, channels, **kwargs):
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
    if module_cls is not None:
        return module_cls(channels, **kwargs)
    return None


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten)
    num_pooled_features = num_features * global_pool.feat_mult()
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_pooled_features, num_classes, 1, bias=True)
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting issue
        fc = nn.Linear(num_pooled_features, num_classes, bias=True)
    return global_pool, fc
#########################################################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x



def drop_blocks(drop_block_rate=0.):
    return [
        None, None,
        DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
        DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]

def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net
    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering
    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.
    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample
    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled
    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled
    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width//4 * 6, stem_width * 2
          * 'deep_tiered_narrow' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(ResNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if 'narrow' in stem_type else 6 * (stem_width // 4)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs_1),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_2, inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem Pooling
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=inplanes, stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def make_resnet_200d():
    net = ResNet(
        Bottleneck,
        layers=[3, 24, 36, 3],
        num_classes=1000,
        in_chans=3,
        cardinality=1,
        base_width=64,
        stem_width=32,
        stem_type='deep',
        output_stride=32,
        block_reduce_first=1,
        down_kernel_size=1,
        avg_down=True,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        aa_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.,
        drop_block_rate=0.,
        global_pool='avg',
        zero_init_last_bn=True,
        block_args=None
    )
    return net

PRETRAIN_CHECKPOINT = '../pretrain_model/resnet200d_ra2-bdba9bf9.pth'

#############################################################################




# def _create_resnet(variant, pretrained=False, **kwargs):
#     return build_model_with_cfg(
#         ResNet, variant, default_cfg=default_cfgs[variant], pretrained=True, **kwargs)
#
# def resnet200d_320(pretrained=False, **kwargs):
#     """Constructs a ResNet-200-D model. NOTE: Duplicate of 200D above w/ diff default cfg for 320x320.
#     """
#     model_args = dict(
#         block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
#     return _create_resnet('resnet200d_320', pretrained, **model_args)


def run_check_pretrain_net():

    net = make_resnet_200d()
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
            #image = image[16:16+224,16:16+224]


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

'''
<All keys matched successfully>
great_white_shark (256, 256, 3)
[8.8864450e-05 9.3973758e-05 8.2875752e-01 3.0844772e-02 1.4558190e-03]
0        great white shark   2 0.8287575
1              tiger shark   3 0.030844772
2               hammerhead   4 0.001455819
3                 sturgeon 394 0.0005560306
4                    boxer 242 0.0005085576

screwdriver (256, 256, 3)
[0.00015108 0.00013629 0.00022238 0.00030125 0.00030415]
0              screwdriver 784 0.83083063
1                    screw 783 0.0087825125
2          carpenter's kit 477 0.0015460051
3                  syringe 845 0.000982948
4                    cairn 192 0.00086218264

ostrich (256, 256, 3)
[0.00010937 0.00013641 0.00024249 0.00023203 0.00038676]
0                  ostrich   9 0.8458243
1                  bustard 138 0.0038598555
2                    zebra 340 0.0009269517
3               pencil box 709 0.00073487643
4                   pickup 717 0.0006607857

blad_eagle (256, 256, 3)
[0.00013825 0.00012043 0.00017717 0.00019122 0.00021146]
0               bald eagle  22 0.8303348
1                     kite  21 0.01475857
2                  vulture  23 0.0013159747
3                bottlecap 455 0.0012336351
4                albatross 146 0.0011204442

english_foxhound (256, 256, 3)
[0.00029532 0.00018478 0.00042541 0.00031861 0.00029447]
0         English foxhound 167 0.5626837
1             Walker hound 166 0.17052484
2                   beagle 162 0.006274665
3                 bluetick 164 0.0032574516
4            Saint Bernard 247 0.0029420415

goldfish (256, 256, 3)
[7.6774304e-04 8.0734766e-01 1.8053305e-04 1.8162670e-04 2.7908958e-04]
0                 goldfish   1 0.80734766
1                    tench   0 0.00076774304
2                 flamingo 130 0.0007272631
3            measuring cup 647 0.00066795264
4               greenhouse 580 0.0006451642


Process finished with exit code 0

'''
if __name__ == '__main__':
    run_check_pretrain_net()
