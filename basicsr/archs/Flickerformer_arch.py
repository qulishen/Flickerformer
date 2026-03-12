## SCAM2, encoder部分只处理每一张图片，分别处理，然后每一个module使用PAM+FFN用来过滤信息，最后再concatenate三帧后，开始decoder。这样的故事也能讲的通，就是encoder用来筛选信息。最后的decoder用来融合。

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import torch.nn.init as init
from basicsr.utils.registry import ARCH_REGISTRY
import math
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
##########################################################################
## Layer Norm

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class DFFN_AutoCorr(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN_AutoCorr, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        # 自相关融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制频域融合强度
        self.beta = nn.Parameter(torch.tensor(0.5))   # 控制空间域融合强度

    def forward(self, x):
        x = self.project_in(x)

        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        # FFT
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft
        # 自相关功率谱
        power = Xf * torch.conj(Xf)          # |X|^2
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        # 融合（频域 + 空间域）
        Xf_new = Xf + self.alpha * power     # 频域增强周期结构
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))
        x_patch_new = x_patch_new + self.beta * R  # 空间域增强

        # 重组
        x = rearrange(
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output
    
class DWT_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias = False):
        super(DWT_Attention,self).__init__()
        self.dwt = DWTForward(J=1, wave='haar')  # 一层 Haar 小波变换
        self.high_conv = nn.Sequential(
                nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1, groups=2 ,bias= bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),
                nn.ReLU(inplace=True)
                # nn.Sigmoid()
            )
        self.high_out = nn.Sequential(
                nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=3, bias= bias),
                nn.ReLU(inplace=True),
            )
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.idwt = DWTInverse(wave='haar')
        # self.idwt = IWT()
        self.patch_size = 8
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        LL, Yh = self.dwt(x)   # LL: (B,C,H/2,W/2), Yh[0]: (B,C,3,H/2,W/2)]
        Yh = Yh[0]
        LH = Yh[:, :, 0, :, :]
        HL = Yh[:, :, 1, :, :]
        HH = Yh[:, :, 2, :, :]

        filter_hv = self.high_conv(torch.cat([LH,HL],dim=1))

        q,k,v = self.qkv_dwconv(self.qkv(LL)).chunk(3,dim=1)
        b,c,h,w = q.shape

        v = v * filter_hv + v

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        Yh = self.high_out(torch.cat([LH,HL,HH],dim=1))

        LH, HL, HH = Yh.chunk(3,dim=1)

        Yh = torch.stack([LH, HL, HH],dim=2)

        x_hat = self.idwt((out,[Yh]))

        return x_hat 

class DWT_WindowAttention_SW(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, window_size=8, shift_size=4, bias=False):
        super(DWT_WindowAttention_SW, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size
        if (input_resolution//2)  <= window_size: # wavelet need /2
            self.shift_size = 0
            self.window_size = input_resolution //2
            window_size = input_resolution //2

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # DWT
        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')

        # 高频信息卷积
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True)
        )
        self.high_out = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True)
        )

        # QKV for low-frequency attention
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 相对位置偏置（Swin Transformer style）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size)))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    # ========================= 工具函数 =========================
    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.view(B, C, H//ws, ws, W//ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, ws, ws)
        return x

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        C = windows.shape[1]
        ws = self.window_size
        x = windows.view(B, H//ws, W//ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        return x

    def shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        return x

    def reverse_shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        return x

    def window_attention(self, q, k, v):
        """
        q,k,v: (B_win, num_heads, head_dim, N)
        返回: (B_win, num_heads, head_dim, N)
        """
        # q^T * k -> (B_win, num_heads, N, N)
        q = F.normalize(q, dim=-2)  # 沿 head_dim 归一化
        k = F.normalize(k, dim=-2)
        attn = torch.matmul(q.transpose(-2, -1), k)  # (B_win, head, N, N)

        # 相对位置偏置
        N = self.window_size * self.window_size
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2,0,1).unsqueeze(0)  # (1, head, N, N)
        attn = attn + relative_position_bias

        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        # 注意力矩阵乘 v -> (B_win, head, head_dim, N)
        out = torch.matmul(v, attn.transpose(-2, -1))  # (B_win, head, head_dim, N)
        return out


    def forward(self, x):
        B, C, H, W = x.shape

        # ----------- DWT -----------
        LL, Yh = self.dwt(x)
        Yh = Yh[0]
        LH, HL, HH = Yh[:, :, 0, :, :], Yh[:, :, 1, :, :], Yh[:, :, 2, :, :]

        # ----------- 高频信息卷积 -----------
        filter_hv = self.high_conv(torch.cat([LH, HL], dim=1))

        # ----------- QKV & 加权 V -----------
        qkv = self.qkv_dwconv(self.qkv(LL))
        q, k, v_inp = qkv.chunk(3, dim=1)
        v = v_inp * filter_hv + v_inp

        # ----------- Shifted Window Attention -----------
        x_shifted = self.shift(LL, self.shift_size)
        q = self.window_partition(x_shifted)
        k = self.window_partition(x_shifted)
        v = self.window_partition(v)

        # reshape for multi-head
        B_win, Cq, ws, _ = q.shape
        q = q.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)
        k = k.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)
        v = v.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)

        out = self.window_attention(q, k, v)
        out = out.view(B_win, Cq, ws, ws)
        out = self.window_reverse(out, H//2, W//2)
        out = self.reverse_shift(out, self.shift_size)
        out = self.project_out(out)

        # ----------- 高频信息重建 -----------
        Yh = self.high_out(torch.cat([LH, HL, HH], dim=1))
        LH, HL, HH = Yh.chunk(3, dim=1)
        Yh = torch.stack([LH, HL, HH], dim=2)
        x_hat = self.idwt((out, [Yh]))

        return x_hat



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, ffn_expansion_factor, bias, LayerNorm_type, use_att = True):
        super(TransformerBlock, self).__init__()
        # self.att = StripAttention(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        # self.att = Attention(dim, num_heads, bias)
        # self.att = FSAS(dim, bias)
        self.use_att = use_att
        if use_att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.att = DWT_WindowAttention_SW(dim, num_heads, input_resolution, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)
        self.ffn = DFFN_AutoCorr(dim, ffn_expansion_factor, bias)
        

    def forward(self, x):
        if self.use_att:
            x = x + self.att(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class PAM(nn.Module):
    ## gated操作，输出是频闪需要注意的区域权重
    def __init__(self, dim, expand = 2, LayerNorm_type = 'WithBias'):
        super(PAM, self).__init__()
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.process1 = nn.Sequential(
            nn.Conv2d(dim, expand * dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand * dim, dim, 1, 1, 0))
        
    def forward(self, x):
        _, _, H, W = x.shape
        x_norm = self.norm(x)
        x_freq = torch.fft.rfft2(x_norm, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        pha = self.process1(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class Encoder(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Encoder, self).__init__()
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = self.ffn(x) + x 
        return x

class Add(nn.Module):
    def __init__(self, runtime = 'mtk'):
        super(Add, self).__init__()
        self.runtime = runtime
    def forward(self, x0, x1):
        out = x0 + x1
        return out
@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, runtime='mtk'):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.add = Add(runtime=runtime)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        # return identity + out
        return self.add(identity, out)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


## 这个用来融合三个通道特征
class FFM(nn.Module):
    def __init__(self, dim):
        super(FFM, self).__init__()
        self.pam0 = PAM(dim)
        self.pam1 = PAM(dim)
        self.pam2 = PAM(dim)

        self.fusion = nn.Sequential(nn.Conv2d(dim*3, dim, kernel_size=3,stride=1,padding=1),
                        nn.ReLU())
        # self.dwconv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1),
        #                 nn.ReLU())
        
    def forward(self, x0, x1 ,x2):

        x0 = x0 * self.pam0(x0)
        x1 = x1 * self.pam1(x1)
        x2 = x2 * self.pam2(x2)

        x = self.fusion(torch.cat([x0,x1,x2],dim=1))
        # ## gated机制
        # x_1, x_2 = x.chunk(2, dim = 1)
        # x_1 = self.dwconv(x_1)
        # x = x_1 * x_2
        return x

class SimpleBlockReLU(nn.Module):
    def __init__(self, depth=3, input_channels=3, output_channels=64, kernel_size=3):
        super(SimpleBlockReLU, self).__init__()
        padding = int((kernel_size -1)/2)
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.simple_block = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.simple_block(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SCAM(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.conv1 = nn.Conv2d(dim, 2*dim, kernel_size=3,stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, 3*dim, kernel_size=3,stride=1, padding=1)
        self.conv3 = nn.Conv2d(dim, 2*dim, kernel_size=3,stride=1, padding=1)

        self.fusion = nn.Conv2d(3*dim, dim, kernel_size=3,stride=1, padding=1)
        self.norm = LayerNorm(dim, LayerNorm_type= 'WithBias')
        
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        b,c,h,w = x.shape

        x1,x2,x3= x.chunk(3, dim=1)

        k1,v1 = self.conv1(x1).chunk(2,dim=1)
        q1,q2,x2 = self.conv2(x2).chunk(3,dim=1)
        k2,v2 = self.conv3(x3).chunk(2,dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
 
        attn12 = (q1 @ k1.transpose(-2, -1)) * self.temperature1  ##这个地方可以加一个温度的约束，比如说temperature1 + 2 = 1
        attn23 = (q2 @ k2.transpose(-2, -1)) * self.temperature2

        out1 = (attn12 @ v1)
        out2 = (attn23 @ v2)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.fusion(torch.cat([out1,x2,out2],dim=1))

        out = self.ffn(self.norm(out)) + out

        return out
    
class PhaseGuidedFilter(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2.66, bias=False):

        super(PhaseGuidedFilter,self).__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        self.net12 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1,bias=bias),
            nn.Sigmoid()
        )
        self.net23 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1,bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1,bias=bias),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(dim*3, dim, 3, 1, 1)

        self.group_conv = nn.Conv2d(dim*3, dim*3, 3, 1, 1, groups=3)

        self.eps = float(1e-8)
    
    def forward(self, x):
        x1, x2, x3 = x.chunk(3,dim=1)

        f1 = torch.fft.rfft2(x1)
        f2 = torch.fft.rfft2(x2)
        f3 = torch.fft.rfft2(x3)

        mag_1 = torch.abs(f1)
        mag_2 = torch.abs(f2)
        mag_3 = torch.abs(f3)

        phase1 = f1 / (mag_1 + self.eps)
        phase2 = f2 / (mag_2 + self.eps)
        phase3 = f3 / (mag_3 + self.eps)

        C12 = torch.abs(phase1 * torch.conj(phase2))
        C23 = torch.abs(phase3 * torch.conj(phase2))

        C12 = self.net12(C12)
        C23 = self.net23(C23)

        f1_filtered = C12 * f1
        f3_filtered = C23 * f3

        x1_filtered = torch.fft.irfft2(f1_filtered)
        x3_filtered = torch.fft.irfft2(f3_filtered)

        out = torch.cat([x1_filtered,x2,x3_filtered], dim=1)

        out = self.fusion(out)

        return out 
        
 ##########################################################################
#---------- Restormer -----------------------
@ARCH_REGISTRY.register()
class Flickerformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        img_size= 512,
        dim = 32,
        num_blocks = [2,2,2,2], 
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Flickerformer, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        
        self.fusion  = PhaseGuidedFilter(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**0), num_heads=heads[0], input_resolution = img_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_att=False) for i in range(num_blocks[0])])
        
        self.down1_2= Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], input_resolution = img_size//2 , ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_att=False) for i in range(num_blocks[1])])

        self.down2_3 =  Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], input_resolution = img_size//4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_att=False) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], input_resolution = img_size//4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], input_resolution = img_size//2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], input_resolution = img_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        x1, x2, x3 = inp_img.chunk(3, dim=1)
        inp_enc_level1_x1 = self.conv1(x1)
        inp_enc_level1_x2 = self.conv2(x2)
        inp_enc_level1_x3 = self.conv3(x3)

        out_enc_level1_x2 = self.fusion(torch.cat([inp_enc_level1_x1,inp_enc_level1_x2,inp_enc_level1_x3],dim=1))

        out_enc_level1_x2 = self.encoder_level1(out_enc_level1_x2)

        inp_enc_level2 = self.down1_2(out_enc_level1_x2)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = torch.cat([inp_dec_level2,out_enc_level2], dim = 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1,out_enc_level1_x2],dim=1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1 + x2


if __name__ == "__main__":
    net = Flickerformer().cuda()
    # x = torch.randn(1, 9, 4096, 3072)
    x = torch.randn(1, 9, 512, 512).cuda() #  
    
    input_all = x
    x = net(input_all)
    print(x.shape)

    from thop import profile
    macs, params = profile(net, inputs=(input_all,))
    print("flops: {:.2f} GFLOPs, params: {:.2f} M".format(macs / 1e9, params / 1e6))
    