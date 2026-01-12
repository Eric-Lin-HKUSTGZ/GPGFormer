from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        q = q.view(B, N, self.heads, -1).transpose(1, 2)  # (B, H, N, Dh)
        k = k.view(B, N, self.heads, -1).transpose(1, 2)
        v = v.view(B, N, self.heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,N,Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B,N,inner)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, Nq, dim), context: (B, Nk, context_dim)
        B, Nq, _ = x.shape
        Nk = context.shape[1]

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = kv

        q = q.view(B, Nq, self.heads, -1).transpose(1, 2)  # (B,H,Nq,Dh)
        k = k.view(B, Nk, self.heads, -1).transpose(1, 2)  # (B,H,Nk,Dh)
        v = v.view(B, Nk, self.heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,Nq,Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,Nq,Dh)
        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.to_out(out)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.self_attn = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.cross_attn = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    """
    Minimal cross-attention Transformer decoder:
      - self-attention on query tokens
      - cross-attention to context tokens
      - MLP
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int = 1280,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if skip_token_embedding:
            if token_dim != dim:
                raise ValueError(f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding=True")
            self.to_token_embedding = nn.Identity()
        else:
            self.to_token_embedding = nn.Linear(token_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim=dim,
                    context_dim=context_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, inp: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # inp: (B, num_tokens, token_dim); context: (B, Nk, context_dim)
        x = self.to_token_embedding(inp)
        x = self.dropout(x)
        x = x + self.pos_embedding[:, : x.shape[1]]
        for layer in self.layers:
            x = layer(x, context=context)
        return x










import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        q = q.view(B, N, self.heads, -1).transpose(1, 2)  # (B, H, N, Dh)
        k = k.view(B, N, self.heads, -1).transpose(1, 2)
        v = v.view(B, N, self.heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,N,Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B,N,inner)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, Nq, dim), context: (B, Nk, context_dim)
        B, Nq, _ = x.shape
        Nk = context.shape[1]

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = kv

        q = q.view(B, Nq, self.heads, -1).transpose(1, 2)  # (B,H,Nq,Dh)
        k = k.view(B, Nk, self.heads, -1).transpose(1, 2)  # (B,H,Nk,Dh)
        v = v.view(B, Nk, self.heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,Nq,Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,Nq,Dh)
        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.to_out(out)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.self_attn = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.cross_attn = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    """
    Minimal cross-attention Transformer decoder:
      - self-attention on query tokens
      - cross-attention to context tokens
      - MLP
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int = 1280,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if skip_token_embedding:
            if token_dim != dim:
                raise ValueError(f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding=True")
            self.to_token_embedding = nn.Identity()
        else:
            self.to_token_embedding = nn.Linear(token_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim=dim,
                    context_dim=context_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, inp: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # inp: (B, num_tokens, token_dim); context: (B, Nk, context_dim)
        x = self.to_token_embedding(inp)
        x = self.dropout(x)
        x = x + self.pos_embedding[:, : x.shape[1]]
        for layer in self.layers:
            x = layer(x, context=context)
        return x









