import torch
from torch import nn

class STRAM(nn.Module):
    def __init__(self, args, dim_in, hidden_dim):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim):
        dropout = args.merger_dropout

        self.linear_1 = nn.Linear(4, dim_in)
        self.linear_2 = nn.Linear(4, dim_in)
        self.linear_3 = nn.Linear(4, dim_in)
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

    def forward(self, trajectories, marks, humans):
        trajectories = self.linear_1(trajectories)
        marks = self.linear_2(marks)
        humans = self.linear_3(humans)
        q_oe = k_oe = v_oe = torch.cat([trajectories, humans, marks], 1)

        tgt_oe = v_oe
        tgt2_oe = self.self_attn_oe(q_oe, k_oe, value=tgt_oe)[0]
        tgt_oe = tgt_oe + self.dropout1_oe(tgt2_oe)
        tgt_oe = self.norm1_oe(tgt_oe)

        tgt2_oe = self.linear2_oe(self.dropout_oe(self.activation(self.linear1_oe(tgt_oe))))
        # add & norm
        tgt_oe = tgt_oe + self.dropout2_oe(tgt2_oe)
        tgt_oe = self.norm2_oe(tgt_oe)

        trajectories_len = trajectories.shape[1]
        marks_len = marks.shape[1]
        return tgt_oe[:, :trajectories_len, :], tgt_oe[:, trajectories_len:trajectories_len+marks_len, :], tgt_oe[:, trajectories_len+marks_len:, :]

def build(args):
    return STRAM(args, args.d_model, args.hidden_dim)