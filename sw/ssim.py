# The code in this file has been referenced from
# https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py


import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    """Create 1-D gauss kernel

    Args:
        size (int): The size of gauss kernel.
        sigma (float): Sigma of normal distribution.
    
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    """Blur input with 1-D kernel

    Args:
        input (torch.Tensor): A batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    
    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape

    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = F.conv2d(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=out.shape[1])
        else:
            warnings.warn(f'Skipping Gaussian Smoothing at dim 2 + {i} for input: {input.shape} and win size: {win.shape[-1]}')

    return out


def _ssim(
    X, Y,
    data_range,
    win,
    size_average=True,
    K=(0.01, 0.03)
):
          
    """Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): Images
        Y (torch.Tensor): Images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int): Value range of input images
            (usually 1.0 or 255).
        size_average (bool, optional): If size_average=True,
            ssim of all images will be averaged as a scalar.
            (default: True)

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)
    
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12   = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten( cs_map, 2 ).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False
):
    """Interface for ssim

    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): Value range of input
            images (usually 1.0 or 255). (default: 255)
        size_average (bool, optional): If size_average=True, ssim
            of all images will be averaged as a scalar. (default: True)
        win_size: (int, optional): The size of gauss kernel.
            (default: 11)
        win_sigma: (float, optional): Sigma of normal distribution.
            (default: 1.5)
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new
            kernel will be created according to win_size and win_sigma.
            (default: None)
        K (list or tuple, optional): Scalar constants (K1, K2). Try a
            larger K2 constant (e.g. 0.4) if you get a negative or NaN
            results. (default: (0.01, 0.03))
        nonnegative_ssim (bool, optional): Force the ssim response to be
            nonnegative with relu.
    
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same shape.')
    
    if win is not None: # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')
    
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    
    ssim_per_channel, cs = _ssim(
        X, Y,
        data_range=data_range,
        win=win,
        size_average=False,
        K=K
    )
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)
    
    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    weights=None,
    K=(0.01, 0.03)
):

    """Interface for ms-ssim

    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): Value range of input
            images (usually 1.0 or 255). (default: 255)
        size_average (bool, optional): If size_average=True, ssim
            of all images will be averaged as a scalar. (default: True)
        win_size: (int, optional): The size of gauss kernel.
            (default: 11)
        win_sigma: (float, optional): Sigma of normal distribution.
            (default: 1.5)
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new
            kernel will be created according to win_size and win_sigma.
            (default: None)
        weights (list, optional): Weights for different levels.
            (default: None)
        K (list or tuple, optional): Scalar constants (K1, K2). Try a
            larger K2 constant (e.g. 0.4) if you get a negative or NaN
            results. (default: (0.01, 0.03))
    
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same dimensions.')
    
    if win is not None: # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    smaller_side = min( X.shape[-2:] )
    assert smaller_side > (win_size-1) * (2**4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim"% ((win_size-1) * (2**4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             K=K)

        if i < levels-1: 
            mcs.append(torch.relu(cs))
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)
    
    ssim_per_channel = torch.relu(ssim_per_channel) # (batch, channel)
    mcs_and_ssim = torch.stack(mcs+[ssim_per_channel], dim=0) # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=1.0,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        K=(0.01, 0.03),
        nonnegative_ssim=False
    ):
        """Class for ssim

        Args:
            data_range (float or int, optional): Value range of input
                images (usually 1.0 or 255). (default: 1.0)
            size_average (bool, optional): If size_average=True, ssim
                of all images will be averaged as a scalar. (default: True)
            win_size: (int, optional): The size of gauss kernel.
                (default: 11)
            win_sigma: (float, optional): Sigma of normal distribution.
                (default: 1.5)
            channel (int, optional): Input channels (default: 3)
            K (list or tuple, optional): Scalar constants (K1, K2). Try a
                larger K2 constant (e.g. 0.4) if you get a negative or NaN
                results. (default: (0.01, 0.03))
            nonnegative_ssim (bool, optional): Force the ssim response to be
                nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X, Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim
        )


class MSSSIM(torch.nn.Module):

    def __init__(
        self,
        data_range=1.0,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        weights=None,
        K=(0.01, 0.03)
    ):
        """Class for ms-ssim

        Args:
            data_range (float or int, optional): Value range of input
                images (usually 1.0 or 255). (default: 1.0)
            size_average (bool, optional): If size_average=True, ssim
                of all images will be averaged as a scalar. (default: True)
            win_size: (int, optional): The size of gauss kernel.
                (default: 11)
            win_sigma: (float, optional): Sigma of normal distribution.
                (default: 1.5)
            channel (int, optional): input channels (default: 3)
            weights (list, optional): Weights for different levels.
                (default: None)
            K (list or tuple, optional): Scalar constants (K1, K2). Try a
                larger K2 constant (e.g. 0.4) if you get a negative or NaN
                results. (default: (0.01, 0.03))
        """

        super(MSSSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X, Y, 
            data_range=self.data_range, 
            size_average=self.size_average, 
            win=self.win, 
            weights=self.weights, 
            K=self.K
        )


class SSIMLoss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIMLoss, self).forward(img1, img2)


class MSSSIMLoss(MSSSIM):
    def forward(self, img1, img2):
        return 1 - super(MSSSIMLoss, self).forward(img1, img2)
