import argparse

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_adaptive_avgpool2d(m, x, y):
    x = x[0]
    num_elements = x.numel()
    output_elements = y.numel()
    kernel_size = num_elements // output_elements
    total_add = kernel_size - 1
    total_div = 1
    total_ops = output_elements * (total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_upsample_bilinear2d(m, x, y):
    output_elements = y.numel()
    total_ops = output_elements * 7  # 4 muls + 3 adds

    m.total_ops += torch.Tensor([int(total_ops)])

def count_identity(m, x, y):
    m.total_ops += torch.Tensor([0])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            m.register_forward_hook(count_adaptive_avgpool2d)
        elif isinstance(m, nn.UpsamplingBilinear2d):
            m.register_forward_hook(count_upsample_bilinear2d)
        elif isinstance(m, nn.Identity):
            m.register_forward_hook(count_identity)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params
    total_ops = total_ops
    total_params = total_params

    return total_ops, total_params

def main():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=2,
        classes=1,
    )
    model.load_state_dict(torch.load("/kaggle/input/deeplabv3_ckpt_epoch_18_88/other/default/1/deeplabv3_ckpt_epoch_18.pth"))
    total_ops, total_params = profile(model, (4,2,256,256))
    print("#Ops: %f GOps"%(total_ops/1e9))
    print("#Parameters: %f M"%(total_params/1e6))

if __name__ == "__main__":
    main()
