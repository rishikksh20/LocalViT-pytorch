import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from module import Residual, Attention, PreNorm, ConvFF
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, patch_height, patch_width, scale = 4, depth_kernel = 3, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, ConvFF(dim, scale, depth_kernel, patch_height, patch_width)))
            ]))
    def forward(self, x):

        for attn, convff in self.layers:
            x = attn(x)
            cls_tokens = x[:, 0]
            x = convff(x[:, 1:])
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1) 
        return x


  
class LocalViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim = 192, depth = 12, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., depth_kernel = 3, scale_dim = 4):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        patch_height = int(image_size//patch_size)
        patch_width = int(image_size//patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, patch_height, patch_width, scale_dim, depth_kernel, dropout)
        


        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 3, 224, 224])
    
    model = LocalViT(224, 16, 100)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)