"""Defines the conditional discrete diffusion network and feature-fusion modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L]


class TimeEmbedding(nn.Module):

    def __init__(self, d_model, max_steps=10000):
        super().__init__()
        table = SinusoidalPositionalEncoding(d_model, max_steps).pe.squeeze(0)
        self.register_buffer("table", table)

    def forward(self, t, L):
        t = t.clamp(0, self.table.size(0) - 1)
        pe = self.table.index_select(0, t)
        return pe.unsqueeze(1).expand(-1, L, -1)


class TransformerBackbone(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        return self.enc(x)


class CrossAttentionBlock(nn.Module):


    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h, cond_seq):


        if cond_seq is None:
            return h
        out, _ = self.attn(h, cond_seq, cond_seq)
        return self.ln(h + out)


class GateFusion(nn.Module):


    def __init__(self, d_model=256, channel_gate=False):
        super().__init__()
        self.channel_gate = channel_gate
        in_dim = 2 * d_model
        if channel_gate:
            self.gate = nn.Linear(in_dim, d_model)
        else:
            self.gate = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()


        self.last_gate = None

    def forward(self, h_global, h_local):

        if h_global is None and h_local is None:
            self.last_gate = None
            return None
        if h_global is None:
            self.last_gate = None
            return h_local
        if h_local is None:
            self.last_gate = None
            return h_global


        z = torch.cat([h_global, h_local], dim=-1)
        g = self.sigmoid(self.gate(z))


        self.last_gate = g.detach()

        return g * h_local + (1.0 - g) * h_global


class D3PM(nn.Module):
    def __init__(
        self,
        vocab_size=4,
        mask_token_id=4,
        seq_len=165,
        d_model=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,


        use_global=False,
        use_full_local=False,
        use_motif_local=False,


        global_mode="concat",
        local_mode="add",


        fusion_mode="auto",


        cross_heads=8,
        cross_dim=256,
        cross_layers=1,


        gate_channel=False,


        motif_span_left=35,
        motif_span_right=7,
    ):
        super().__init__()


        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.seq_len = seq_len
        self.d_model = d_model

        self.use_global = use_global
        self.use_full_local = use_full_local
        self.use_motif_local = use_motif_local

        self.global_mode = global_mode
        self.local_mode = local_mode
        self.fusion_mode = fusion_mode

        self.motif_span_left = motif_span_left
        self.motif_span_right = motif_span_right


        self.token_emb = TokenEmbedding(vocab_size + 1, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, seq_len + 32)
        self.time_emb = TimeEmbedding(d_model)


        if use_global:
            if global_mode == "concat":
                self.global_proj = nn.Linear(d_model + 256, d_model)
                self.gamma = None
                self.beta = None
            elif global_mode == "film":
                self.global_proj = None
                self.gamma = nn.Linear(256, d_model)
                self.beta = nn.Linear(256, d_model)
            else:
                raise ValueError(f"Unknown global_mode: {global_mode}")
        else:
            self.global_proj = None
            self.gamma = None
            self.beta = None


        self.has_any_local = bool(use_full_local or use_motif_local)

        if self.has_any_local:
            if local_mode == "concat":
                self.local_proj = nn.Linear(d_model + 256, d_model)
                self.local_add_proj = None
            elif local_mode == "replace":
                self.local_proj = nn.Linear(256, d_model)
                self.local_add_proj = None
            elif local_mode == "add":

                self.local_proj = None
                self.local_add_proj = nn.Linear(256, d_model) if d_model != 256 else None
            else:
                raise ValueError(f"Unknown local_mode: {local_mode}")
        else:
            self.local_proj = None
            self.local_add_proj = None


        if fusion_mode == "cross_attn":
            self.cross_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, num_heads=cross_heads)
                for _ in range(cross_layers)
            ])

            self.global_cond_proj = nn.Linear(256, d_model) if use_global else None
            self.full_local_cond_proj = nn.Linear(256, d_model) if use_full_local else None
            self.motif_local_cond_proj = nn.Linear(256, d_model) if use_motif_local else None
        else:
            self.cross_blocks = nn.ModuleList()
            self.global_cond_proj = None
            self.full_local_cond_proj = None
            self.motif_local_cond_proj = None


        if fusion_mode == "gate":
            self.gate_fusion = GateFusion(
                d_model=d_model,
                channel_gate=gate_channel,
            )
        else:
            self.gate_fusion = None


        self.backbone = TransformerBackbone(d_model, num_layers, num_heads, dropout)
        self.num_layers = num_layers

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size + 1)


    def _motif_index(self, L: int):
        start = L - self.motif_span_left
        end = L - self.motif_span_right + 1
        return start, end


    def _apply_global(self, h, cond_global):
        if cond_global is None or (not self.use_global):
            return h

        if self.global_mode == "concat":
            B, L, D = h.shape
            g = cond_global.unsqueeze(1).expand(-1, L, -1)
            out = torch.cat([h, g], dim=-1)
            return self.global_proj(out)
        elif self.global_mode == "film":
            gamma = self.gamma(cond_global).unsqueeze(1)
            beta = self.beta(cond_global).unsqueeze(1)
            return h * (1 + gamma) + beta

        return h


    def _merge_local_block(self, h_block, cond_block):


        if self.local_mode == "add":
            if self.local_add_proj is not None:
                cond_block = self.local_add_proj(cond_block)
            else:

                if h_block.size(-1) != cond_block.size(-1):
                    raise ValueError(
                        f"add mode requires matching dimensions: h_block D={h_block.size(-1)}, cond D={cond_block.size(-1)}"
                    )
            return h_block + cond_block

        elif self.local_mode == "replace":
            return self.local_proj(cond_block)

        elif self.local_mode == "concat":
            out = torch.cat([h_block, cond_block], dim=-1)
            return self.local_proj(out)

        raise ValueError(f"Unknown local_mode: {self.local_mode}")


    def _apply_full_local(self, h, cond_full_local):
        if cond_full_local is None or (not self.use_full_local):
            return h

        return self._merge_local_block(h, cond_full_local)


    def _apply_motif_local(self, h, cond_motif_local):
        if cond_motif_local is None or (not self.use_motif_local):
            return h

        B, L, D = h.shape
        start, end = self._motif_index(L)
        h_block = h[:, start:end, :]
        merged = self._merge_local_block(h_block, cond_motif_local)

        h_out = h.clone()
        h_out[:, start:end, :] = merged
        return h_out


    def _apply_cross_attn(self, h, cond_global, cond_full_local, cond_motif_local):
        if self.fusion_mode != "cross_attn":
            return h

        cond_tokens = []

        if self.use_global and (cond_global is not None):
            if self.global_cond_proj is None:
                raise RuntimeError("fusion_mode=cross_attn but global_cond_proj was not created")
            g = self.global_cond_proj(cond_global).unsqueeze(1)
            cond_tokens.append(g)

        if self.use_full_local and (cond_full_local is not None):
            if self.full_local_cond_proj is None:
                raise RuntimeError("fusion_mode=cross_attn but full_local_cond_proj was not created")
            fl = self.full_local_cond_proj(cond_full_local)
            cond_tokens.append(fl)

        if self.use_motif_local and (cond_motif_local is not None):
            if self.motif_local_cond_proj is None:
                raise RuntimeError("fusion_mode=cross_attn but motif_local_cond_proj was not created")
            ml = self.motif_local_cond_proj(cond_motif_local)
            cond_tokens.append(ml)

        if len(cond_tokens) == 0:
            return h

        cond_seq = torch.cat(cond_tokens, dim=1)

        for blk in self.cross_blocks:
            h = blk(h, cond_seq)

        return h


    def predict_logits(self, xt, t, cond_global=None, cond_full_local=None, cond_motif_local=None):


        B, L = xt.shape
        xt = xt.clamp(min=0, max=self.vocab_size)

        tok = self.token_emb(xt)
        h = self.pos(tok)
        time = self.time_emb(t, L=L)
        h = h + time


        if (not self.use_global) and (not self.has_any_local):
            h = self.backbone(h)
            h = self.norm(h)
            logits = self.head(h)
            return logits


        if self.fusion_mode == "auto":

            if self.use_global:
                h = self._apply_global(h, cond_global)
            if self.use_full_local:
                h = self._apply_full_local(h, cond_full_local)
            if self.use_motif_local:
                h = self._apply_motif_local(h, cond_motif_local)
            h = self.backbone(h)

        elif self.fusion_mode == "cross_attn":
            h = self._apply_cross_attn(h, cond_global, cond_full_local, cond_motif_local)
            h = self.backbone(h)

        elif self.fusion_mode == "gate":

            h_g = self._apply_global(h, cond_global) if self.use_global else None

            if self.has_any_local:
                h_l = h
                if self.use_full_local:
                    h_l = self._apply_full_local(h_l, cond_full_local)
                if self.use_motif_local:
                    h_l = self._apply_motif_local(h_l, cond_motif_local)
            else:
                h_l = None

            h_fused = self.gate_fusion(h_g, h_l) if self.gate_fusion is not None else None
            if h_fused is None:
                h_fused = h
            h = self.backbone(h_fused)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")


        h = self.norm(h)
        logits = self.head(h)
        return logits
