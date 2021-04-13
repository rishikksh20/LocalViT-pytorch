from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from math import sqrt

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

    
class ConvFF(nn.Module):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3, patch_height = 14, patch_width = 14, dropout=0.):
        super().__init__()
        
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(
                                    Rearrange('b (h w) c -> b c h w', h=patch_height, w=patch_width),
                                    nn.Conv2d(dim, scale_dim, kernel_size=1),
                                    nn.Hardswish()
                                    )
        
        self.depth_conv = nn.Sequential(
                        nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=True),
                        nn.Conv2d(scale_dim, scale_dim, kernel_size=1, bias=True),
                        nn.Hardswish()
                        )
        
        self.down_proj = nn.Sequential(
                                    nn.Conv2d(scale_dim, dim, kernel_size=1),
                                    nn.Dropout(dropout),
                                    Rearrange('b c h w ->b (h w) c')
                                    )
        
    def forward(self, x):

        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
    
    

        
        
        
        

