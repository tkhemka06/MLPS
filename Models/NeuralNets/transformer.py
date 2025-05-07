import torch
import torch.nn as nn

## SGU Block---------------------------------------------------------------------------------
class SGU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim // 2, dim // 2)
        self.out_proj = nn.Linear(dim // 2, dim)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        gated = x1 * torch.sigmoid(self.gate(x2))
        return self.out_proj(gated)

## gMLP Block--------------------------------------------------------------------------------
class gMLPBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.sgu = SGU(dim_out)
        self.ffn = nn.Sequential(
            nn.Linear(dim_out, dim_out * 4),
            nn.GELU(),
            nn.Linear(dim_out * 4, dim_out),
            nn.Dropout(0.5)
        )
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        residual = self.proj(x)
        x = self.sgu(residual)
        x = self.ffn(x)
        x = residual + x
        return self.norm(x)

## Diamond Encoder and Decoder--------------------------------------------------------------
class DiamondEncoder(nn.Module):
    def __init__(self, dims=[32, 64, 128]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(gMLPBlock(dims[i], dims[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DiamondDecoder(nn.Module):
    def __init__(self, dims=[128, 64, 32]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(gMLPBlock(dims[i], dims[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

## COVID Transformer Model---------------------------------------------------------------------
class COVIDTransformer(nn.Module):
    def __init__(self, input_dim=32, output_dim=32):
        super().__init__()
       
        self.encoder = DiamondEncoder()
        
        
        self.decoder = DiamondDecoder()
        
        # Input/output projections
        self.input_proj = nn.Linear(input_dim, 32)
        self.output_proj = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.output_proj(x)



