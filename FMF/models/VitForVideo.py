"""
代码源自 https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py
"""
import math

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, dim, num_patches=196, PEType='tri'):
        super().__init__()
        self.PEType = PEType

        if PEType == 'tri':
            # Compute the positional encodings once in log space.
            pe = torch.zeros(num_patches + 1, dim)
            position = torch.arange(0, num_patches + 1).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2) *
                                 -(math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        elif PEType == 'learnable':
            self.pe = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        else:
            raise ValueError(f'PEType must be "tri" or "learnable", but get {PEType}')

    def forward(self, x):
        if self.PEType == 'tri':
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        else:
            x = x + self.pe
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ClassificationHead(nn.Module):
    def __init__(self, pool, dim, num_classes):
        super().__init__()
        self.pool = pool

        # for video
        self.to_latent1 = nn.Identity()
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # for current
        self.to_latent2 = nn.Identity()
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x1, x2=None):
        x1 = x1.mean(dim=1) if self.pool == 'mean' else x1[:, 0]
        x1 = self.to_latent1(x1)
        out = self.mlp_head1(x1)

        if x2 is not None:
            x2 = x2.mean(dim=1) if self.pool == 'mean' else x2[:, 0]
            x2 = self.to_latent2(x2)
            out = out + self.mlp_head2(x2)

        return out


class ResidualConvUnit(nn.Module):
    # Residual convolution module
    def __init__(self, features, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(features)

        # self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(features)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.bn1(out)
        out = self.conv1(out)

        # out = self.relu(out)
        # out = self.bn2(out)
        # out = self.conv2(out)

        return out + x


class SegmentationHead(nn.Module):
    def __init__(self, dim, t, h, w, num_classes, hidden_dims, additional=True):
        super().__init__()

        self.num_patches = t * h * w
        self.de_patch_embedding = nn.Sequential(
            Rearrange('b (t h w) c -> b (t c) h w', t=t, h=h, w=w),
            nn.Conv2d(in_channels=t * dim, out_channels=dim, kernel_size=1)
        )

        self.additional = additional
        if additional:
            self.de_patch_embedding_dilated = nn.Sequential(
                Rearrange('b (t h w) c -> b (t c) h w', t=t, h=h // 2, w=w // 2),
                nn.Conv2d(in_channels=t * dim, out_channels=dim, kernel_size=1)
            )
            self.decode_dilated = self.decode1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=dim, out_channels=dim,
                                   kernel_size=2, stride=2, padding=0, bias=True),
                ResidualConvUnit(dim)
            )

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=hidden_dims[0],
                               kernel_size=2, stride=2, padding=0, bias=True),
            ResidualConvUnit(hidden_dims[0])
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[0], out_channels=hidden_dims[1],
                               kernel_size=2, stride=2, padding=0, bias=True),
            ResidualConvUnit(hidden_dims[1])
        )
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[1], out_channels=hidden_dims[2],
                               kernel_size=2, stride=2, padding=0, bias=True),
            ResidualConvUnit(hidden_dims[2])
        )
        self.decode_out = nn.Conv2d(in_channels=hidden_dims[2], out_channels=num_classes, kernel_size=1)

    def forward(self, clip_tokens, *args):
        clip_tokens = clip_tokens[:, 1:]  # 舍弃class token

        # b (t h w) c -> b (t c) h w -> b c h w
        seg_map = self.de_patch_embedding(clip_tokens[:, :self.num_patches])
        if self.additional:
            # b (t h/2 w/2) c -> b (t c) h/2 w/2 -> b c h/2 w/2
            seg_map_dilated = self.de_patch_embedding_dilated(clip_tokens[:, self.num_patches:])
            seg_map_dilated = self.decode_dilated(seg_map_dilated)  # -> b c h w
            seg_map = seg_map + seg_map_dilated

        seg_map = self.decode1(seg_map)  # h w -> 2h, 2w
        seg_map = self.decode2(seg_map)  # 2h, 2w -> 4h, 4w
        seg_map = self.decode3(seg_map)  # 4h, 4w -> 8h, 8w
        seg_map = self.decode_out(seg_map)  # c' -> num_cls
        return seg_map


class AdditionalVideoTokens(nn.Module):
    def __init__(self, image_size, image_patch_size, frames, frame_patch_size, dim, channels=3, emb_dropout=0., ):
        super().__init__()
        image_height, image_width = pair(image_size // 2)  # // 2
        patch_height, patch_width = pair(image_patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.image_size = image_size
        self.frames = frames

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, video):
        # 最近邻插值至空间分辨率为1/4
        st_resolution = (self.frames, self.image_size // 2, self.image_size // 2)
        x = F.interpolate(video, st_resolution, mode='nearest')

        x = self.to_patch_embedding(x)

        # for video
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        return x


class AdditionalFrameTokens(nn.Module):
    def __init__(self, image_size, image_patch_size, dim, channels=3, emb_dropout=0., ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.image_size = image_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv3d(in_channels=channels, out_channels=dim, kernel_size=(1, image_patch_size, image_patch_size),
        #               stride=(2, image_patch_size, image_patch_size)),
        #     Rearrange('b c t h w -> b (t h w) c'),
        #     nn.LayerNorm(dim),
        # )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, video):
        # frame = video[:, :, -1]
        frame = torch.median(video, dim=2)[0]
        x = self.to_patch_embedding(frame)

        # for video
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        return x


class ViTForVideo(nn.Module):
    def __init__(self, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 PEType='learnable', task='classification', additional=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert task in ['classification', 'segmentation']

        self.task = task
        self.additional = additional
        self.pool = pool

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv3d(in_channels=channels, out_channels=dim, kernel_size=(1, image_patch_size, image_patch_size),
        #               stride=(2, image_patch_size, image_patch_size)),
        #     Rearrange('b c t h w -> b (t h w) c'),
        #     nn.LayerNorm(dim),
        # )
        self.pos_embedding = PositionalEncoding(dim=dim, num_patches=num_patches, PEType=PEType)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # additional video tokens
        if additional:
            self.get_additional_tokens = AdditionalVideoTokens(
                image_size=image_size, image_patch_size=image_patch_size,
                frames=frames, frame_patch_size=frame_patch_size, dim=dim
            )
            # self.get_additional_tokens = AdditionalFrameTokens(
            #     image_size=image_size, image_patch_size=im,
            #     dim=dim, channels=3, emb_dropout=0.
            # )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        if task == 'classification':
            self.head = ClassificationHead(pool, dim, num_classes)
        else:
            self.head = SegmentationHead(dim, t=frames // frame_patch_size,
                                         h=image_height // patch_height, w=image_width // patch_width,
                                         num_classes=num_classes, hidden_dims=[128, 64, 32], additional=additional)

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        if self.additional:
            # x_addition = self.get_additional_tokens(video[:, :, -1])
            x_addition = self.get_additional_tokens(video)
            x = torch.cat((x, x_addition), dim=1)

        x = self.transformer(x)

        out = self.head(x)

        return out


def main():
    clip = torch.randn((1, 3, 8, 64, 64))
    model = ViTForVideo(
        image_size=64, image_patch_size=8, frames=8, frame_patch_size=2, num_classes=2, dim=96, depth=6, heads=3,
        mlp_dim=4*96, pool='cls', channels=3, dim_head=32, dropout=0., emb_dropout=0., PEType='learnable',
        task='segmentation', additional=True
    )
    # state_dict = model.state_dict()
    # for k, v in state_dict.items():
    #     print(k, v.shape)

    y = model(clip)

    # label = torch.ones([1], dtype=torch.long)
    label = torch.ones((1, 64, 64), dtype=torch.long)
    loss = F.cross_entropy(y, label)
    loss.backward()

    print(y.shape)


if __name__ == '__main__':
    main()
