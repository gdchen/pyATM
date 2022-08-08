# %%
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        self.kernel_size = kernel_size
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        print(kernel)
        print(kernel.shape)

        self.register_buffer('weight', kernel)
        # self.groups = channels

        self.conv_layer = nn.Conv2d(
            channels, channels, self.kernel_size, padding=self.kernel_size // 2,
            padding_mode="circular", bias=False, groups=channels)
        self.conv_layer.weight = nn.Parameter(kernel)

        # weights = torch.tensor(
        #     [1.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, self.kernel_size)

        # if dim == 1:
        #     self.conv = F.conv1d(input, weight=self.weight, groups=self.groups)
        # elif dim == 2:
        #     self.conv = F.conv2d(input, weight=self.weight, groups=self.groups)
        # elif dim == 3:
        #     self.conv = F.conv3d(input, weight=self.weight, groups=self.groups)
        # else:
        #     raise RuntimeError(
        #         'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
        #             dim)
        #     )

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv_layer(input)


def main():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    input_ = torch.rand(100, 3, 50, 50)

    plt.figure(0)
    plt.imshow(input_[0, 0, :].squeeze())
    model = GaussianSmoothing(channels=3, kernel_size=5, sigma=1, dim=2)
    output_ = model(input_)
    plt.figure(1)
    plt.imshow(output_[0, 0, :].squeeze())
    print(output_.shape)
    print(input_.mean(dim=(0, 2, 3)))
    print(output_.mean(dim=(0, 2, 3)))


if __name__ == "__main__":
    main()


# %%
