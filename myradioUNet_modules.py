import torch
import torch.nn as nn


def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        #In conv, the dimension of the output, if the input is H,W, is
        # H+2*padding-kernel +1
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #pooling takes Height H and width W to (H-pool)/pool+1 = H/pool, and floor. Same for W.
        #altogether, the output size is (H+2*padding-kernel +1)/pool.
    )


def convreluT(in_channels, out_channels, kernel=3, padding=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),  # nearest-neighbour upsampling
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNetEncoder(nn.Module):

    def __init__(self,inputs, out_channs=128, base_channs=32, chan_mult=[1, 2, 4, 8]):
        super().__init__()

        self.chan_mult = chan_mult
        self.base_channs = base_channs

        self.layer00 = convrelu(inputs, self.base_channs*self.chan_mult[0], 3, 1,1)
        self.layer0 = convrelu(self.base_channs*self.chan_mult[0], self.base_channs*self.chan_mult[1], 3, 1,2)

        self.layer1 = convrelu(self.base_channs*self.chan_mult[1], self.base_channs*self.chan_mult[2], 5, 2,2)
        self.layer10 = convrelu(self.base_channs*self.chan_mult[2], self.base_channs*self.chan_mult[3], 5, 2,1)

        self.layer2 = convrelu(self.base_channs*self.chan_mult[3], out_channs, 5, 2,2)

    def forward(self, x):
        features = []
        B, C, H, W = x.shape

        layer00 = self.layer00(x)
        features.append(layer00)
        layer0 = self.layer0(layer00)  # (2, 40, 64, 64)
        features.append(layer0)

        layer1 = self.layer1(layer0)  # (2, 40, 32, 32)
        features.append(layer1)
        layer10 = self.layer10(layer1)
        features.append(layer10)

        layer2 = self.layer2(layer10)  # (2, 100, 16, 16)

        return layer2, features[::-1]


class UNetDecoder(nn.Module):

    def __init__(self,inputs, in_chans=128, out_channs=1, base_channs=32, chan_mult=[8, 4, 2, 1], img_size=128, patch_size=8):
        super().__init__()

        self.chan_mult = chan_mult
        self.base_channs = base_channs

        self.conv_up2 = convreluT(in_chans, self.base_channs*self.chan_mult[0], 4, 2)

        self.conv_up10 = convrelu(2*self.base_channs*self.chan_mult[0], self.base_channs*self.chan_mult[1], 3, 1, 1)
        self.conv_up1 = convreluT(2*self.base_channs*self.chan_mult[1], self.base_channs*self.chan_mult[2], 4, 2)

        self.conv_up0 = convreluT(2*self.base_channs*self.chan_mult[2], self.base_channs*self.chan_mult[3], 4, 2)

        self.conv_up00 = convrelu(2*self.base_channs*self.chan_mult[3]+inputs, self.base_channs*self.chan_mult[3], 3, 1,1)
        self.conv_up000 = convrelu(self.base_channs*self.chan_mult[3]+inputs, out_channs, 3, 1,1)


    def forward(self, x, origin_x, features):

        layer10u = self.conv_up2(x)
        layer10u = torch.cat([layer10u, features[0]], dim=1)

        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([layer1u, features[1]], dim=1)
        layer0u = self.conv_up1(layer1u)

        layer0u = torch.cat([layer0u, features[2]], dim=1)
        layer00u = self.conv_up0(layer0u)

        layer00u = torch.cat([layer00u, features[3]], dim=1)
        layer00u = torch.cat([layer00u, origin_x], dim=1)

        layer000u  = self.conv_up00(layer00u)
        layer000u = torch.cat([layer000u,origin_x], dim=1)
        output  = self.conv_up000(layer000u)
        return output


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MyUNet(nn.Module):

    def __init__(self,inputs, out_channs=1):
        super().__init__()
        self.encoder = UNetEncoder(inputs=inputs)
        self.decoder = UNetDecoder(inputs=inputs, out_channs=out_channs)

    def forward(self, x):
        x_origin = x.clone()
        x, features = self.encoder(x)
        output = self.decoder(x, x_origin, features)
        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} device".format(device))
    model = MyUNet(inputs=2).to(device)

    input_tensor = torch.randn(2, 2, 128, 128).to(device)  # (batch_size, channels, height, width)
    outputs1 = model(input_tensor)

    print(outputs1.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")
