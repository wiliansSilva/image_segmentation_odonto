import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

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
        #return self.fn(self.norm(x), **kwargs) # Pre norm - version 1
        return self.norm(self.fn(x, **kwargs))  #Post norm - version 2

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(
            dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding
        )))
        self.mlp_block = Residual(PreNorm(dim, MLP(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class PatchMergingConv(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()

        self.patch_merge = nn.Conv2d(in_channels, out_channels, kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)

    def forward(self, x):
        x = self.patch_merge(x)
        x = x.permute(0, 2, 3, 1)
        return x

class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dim, layers, downscaling_factor, num_heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        
        assert layers % 2 == 0, "Stage layers need to be divisable by 2 for regular and shifted block."

        self.patch_partition = PatchMergingConv(in_channels=in_channels, out_channels=hidden_dim, downscaling_factor=downscaling_factor)
        self.layers = nn.ModuleList([])
        
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dim, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dim * 4, shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dim, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dim * 4, shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)

        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        
        return x.permute(0, 3, 1, 2)

# Window size = 7
# Query dim = 32
# MLP expansion layer = 4

# Swin-T -> C = 96, {2, 2, 6, 2}
# Swin-S -> C = 96, {2, 2, 18, 2}
# Swin-B -> C = 128, {2, 2, 18, 2}
# Swin-L -> C = 192, {2, 2, 18, 2}
class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = 
        self.stage2 = 
        self.stage4 = 

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)

def swin_t(**kwargs):
    return SwinTransformer(hidden_dim=96, layers=(2,2,6,2), heads=(3,6,12,24), **kwargs)

def swin_s(**kwargs):
    return SwinTransformer(hidden_dim=96, layers=(2,2,18,2), heads=(3,6,12,24), **kwargs)

def swin_b(**kwargs):
    return SwinTransformer(hidden_dim=128, layers=(2,2,18,2), heads=(3,6,12,24), **kwargs)

def swin_l(**kwargs):
    return SwinTransformer(hidden_dim=192, layers=(2,2,18,2), heads=(3,6,12,24), **kwargs)