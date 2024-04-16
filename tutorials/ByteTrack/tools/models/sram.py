import torch
from torch import nn

class SRAM(nn.Module):
    def __init__(self, args, dim_in, hidden_dim):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim):
        dropout = args.merger_dropout

        self.linear_1 = nn.Linear(4, dim_in)
        self.linear_2 = nn.Linear(4, dim_in)
        self.self_attn_oe = nn.MultiheadAttention(dim_in, 8, dropout)

        self.linear1_oe = nn.Linear(dim_in, hidden_dim)
        self.dropout_oe = nn.Dropout(dropout)
        self.linear2_oe = nn.Linear(hidden_dim, dim_in)

        self.norm1_oe = nn.LayerNorm(dim_in)
        self.norm2_oe = nn.LayerNorm(dim_in)

        self.dropout1_oe = nn.Dropout(dropout)
        self.dropout2_oe = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, humans, marks):
        humans = self.linear_1(humans)
        marks = self.linear_2(marks)
        q_oe = k_oe = v_oe = torch.cat([humans, marks], 1)

        tgt_oe = v_oe
        tgt2_oe = self.self_attn_oe(q_oe, k_oe, value=tgt_oe)[0]
        tgt_oe = tgt_oe + self.dropout1_oe(tgt2_oe)
        tgt_oe = self.norm1_oe(tgt_oe)

        tgt2_oe = self.linear2_oe(self.dropout_oe(self.activation(self.linear1_oe(tgt_oe))))
        # add & norm
        tgt_oe = tgt_oe + self.dropout2_oe(tgt2_oe)
        tgt_oe = self.norm2_oe(tgt_oe)

        humans_len = humans.shape[1]
        return tgt_oe[:, :humans_len, :], tgt_oe[:, humans_len:, :]

def build(args):
    return SRAM(args, args.d_model, args.hidden_dim)