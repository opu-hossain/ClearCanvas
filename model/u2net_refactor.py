import torch
import torch.nn.functional as nn

import math

__all__ = ['U2NET_full', 'U2NET_lite']


def _upsample_like(x, size):
    """
    Upsamples input tensor x to have the same spatial size as target tensor.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.
        size (Iterable[int]): Target tensor's spatial size.

    Returns:
        torch.Tensor: Upsampled tensor.
    """
    # Upsamples 'x' tensor to have the same spatial size as 'size' tensor
    # using bilinear interpolation.
    return nn.interpolate(x, size=size, mode='bilinear', align_corners=False)


def _size_map(x, height):
    """
    Generate a dictionary of {height: size} for upsampling.

    Args:
        x (torch.Tensor): Input tensor.
        height (int): Number of upsampling stages.

    Returns:
        dict: A dictionary containing the spatial sizes at each upsampling stage.
    """
    # Get the spatial size of the input tensor
    size = list(x.shape[-2:])

    # Initialize the dictionary and compute the spatial sizes for each upsampling stage
    sizes = {}
    for h in range(1, height + 1):
        # Store the current spatial size
        sizes[h] = size

        # Compute the spatial size for the next upsampling stage
        size = [math.ceil(w / 2) for w in size]

    return sizes


class REBNCONV(nn.Module):
    """
    Reversible Bidirectional Residual Convolutional Network block.

    This block consists of a convolutional layer with a batch normalization layer and a ReLU activation function.

    Args:
        in_ch (int, optional): Number of input channels. Defaults to 3.
        out_ch (int, optional): Number of output channels. Defaults to 3.
        dilate (int, optional): Dilation rate for the convolutional layer. Defaults to 1.
    """
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        """
        Initialize the REBNCONV block.
        """
        super(REBNCONV, self).__init__()

        # Convolutional layer with padding and dilation
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        # Batch normalization layer
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        # ReLU activation function
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the REBNCONV block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    """
    Residual Symmetric Upsampling block.

    This block consists of a symmetric encoder-decoder structure with residual connections.

    Args:
        name (str): Name of the block.
        height (int): Height of the block.
        in_ch (int): Number of input channels.
        mid_ch (int): Number of middle channels.
        out_ch (int): Number of output channels.
        dilated (bool, optional): Whether to use dilated convolutions. Defaults to False.
    """
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        """
        Initialize the RSU block.
        """
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        """
        Forward pass of the RSU block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        """
        Make the layers of the RSU block.

        Args:
            height (int): Height of the block.
            in_ch (int): Number of input channels.
            mid_ch (int): Number of middle channels.
            out_ch (int): Number of output channels.
            dilated (bool, optional): Whether to use dilated convolutions. Defaults to False.
        """
        self.add_module('rebnconvin', REBNCONV(in_ch, out_ch))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module(f'rebnconv1', REBNCONV(out_ch, mid_ch))
        self.add_module(f'rebnconv1d', REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(mid_ch, mid_ch, dilate=dilate))


class U2NET(nn.Module):
    """
    U^2-Net model for image segmentation.

    Args:
        cfgs (dict): Configuration for building RSU blocks and side layers.
        out_ch (int): Number of output channels.
    """
    def __init__(self, cfgs, out_ch):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self._make_layers(cfgs)

    def forward(self, x):
        """
        Forward pass of the U^2-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list: List of saliency probability maps.
        """
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            """
            Recursive function for building the U^2-Net model.

            Args:
                x (torch.Tensor): Input tensor.
                height (int): Height of the current block.

            Returns:
                torch.Tensor: Output tensor.
            """
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            """
            Build side output layer.

            Args:
                x (torch.Tensor): Input tensor.
                h (int): Height of the current block.
            """
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            """
            Fuse saliency probability maps.
            """
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        """
        Build the layers of the U^2-Net model.

        Args:
            cfgs (dict): Configuration for building RSU blocks and side layers.
        """
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.out_ch, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1))


def U2NET_full():
    """
    Builds the full version of the U^2-Net model.

    Returns:
        U2NET: The full version of the U^2-Net model.
    """
    # Configurations for building RSUs and sides
    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
    full = {
        'stage1': ['En_1', (7, 3, 32, 64), -1],  # Stage 1: Encoder 1
        'stage2': ['En_2', (6, 64, 32, 128), -1],  # Stage 2: Encoder 2
        'stage3': ['En_3', (5, 128, 64, 256), -1],  # Stage 3: Encoder 3
        'stage4': ['En_4', (4, 256, 128, 512), -1],  # Stage 4: Encoder 4
        'stage5': ['En_5', (4, 512, 256, 512, True), -1],  # Stage 5: Encoder 5 (dilated)
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],  # Stage 6: Encoder 6 (side output)
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],  # Stage 5 Decoder
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],  # Stage 4 Decoder
        'stage3d': ['De_3', (5, 512, 64, 128), 128],  # Stage 3 Decoder
        'stage2d': ['De_2', (6, 256, 32, 64), 64],  # Stage 2 Decoder
        'stage1d': ['De_1', (7, 128, 16, 64), 64],  # Stage 1 Decoder
    }

    # Build the full version of the U^2-Net model
    return U2NET(cfgs=full, out_ch=1)


def U2NET_lite():
    """
    Builds the lite version of the U^2-Net model.

    Returns:
        U2NET: The lite version of the U^2-Net model.
    """
    # Configurations for building RSUs and sides
    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
    lite = {
        # Stage 1: Encoder 1
        'stage1': ['En_1', (7, 3, 16, 64), -1],
        # Stage 2: Encoder 2
        'stage2': ['En_2', (6, 64, 16, 64), -1],
        # Stage 3: Encoder 3
        'stage3': ['En_3', (5, 64, 16, 64), -1],
        # Stage 4: Encoder 4
        'stage4': ['En_4', (4, 64, 16, 64), -1],
        # Stage 5: Encoder 5 (dilated)
        'stage5': ['En_5', (4, 64, 16, 64, True), -1],
        # Stage 6: Encoder 6 (side output)
        'stage6': ['En_6', (4, 64, 16, 64, True), 64],
        # Stage 5 Decoder
        'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
        # Stage 4 Decoder
        'stage4d': ['De_4', (4, 128, 16, 64), 64],
        # Stage 3 Decoder
        'stage3d': ['De_3', (5, 128, 16, 64), 64],
        # Stage 2 Decoder
        'stage2d': ['De_2', (6, 128, 16, 64), 64],
        # Stage 1 Decoder
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }

    # Build the lite version of the U^2-Net model
    return U2NET(cfgs=lite, out_ch=1)
