import torch
import torch.nn as nn
import torch.nn.functional as F



class AlgoExpert(nn.Module):
    def __init__(self, r1 , r2, alpha , d_model):

        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.alpha = alpha
        self.scale = alpha / r2     

        self.P = nn.Parameter(torch.empty(r2, r1))
        self.B = nn.Parameter(torch.zeros(d_model, r2))

        nn.init.kaiming_uniform_(self.P, a=5 ** 0.5)

    def forward(self, shared_out) :
        # [B, T, r1] @ P.T  →  [B, T, r2]
        h = F.linear(shared_out, self.P)         # x @ P.T

        # [B, T, r2] @ B.T  →  [B, T, d_model]
        out = F.linear(h, self.B)                # x @ B.T

        return self.scale * out
    



class SyntaxPolyExpert(nn.Module):
    def __init__(self, r1 , r2, alpha , d_model):

        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.alpha = alpha
        self.scale = alpha / r2     

        self.P = nn.Parameter(torch.empty(r2, r1))
        self.B = nn.Parameter(torch.zeros(d_model, r2))

        nn.init.kaiming_uniform_(self.P, a=5 ** 0.5)

    def forward(self, shared_out) :
        # [B, T, r1] @ P.T  →  [B, T, r2]
        h = F.linear(shared_out, self.P)         # x @ P.T

        # [B, T, r2] @ B.T  →  [B, T, d_model]
        out = F.linear(h, self.B)                # x @ B.T

        return self.scale * out
    



class SecureExpert(nn.Module):
    def __init__(self, r1 , r2, alpha , d_model):

        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.alpha = alpha
        self.scale = alpha / r2     

        self.P = nn.Parameter(torch.empty(r2, r1))
        self.B = nn.Parameter(torch.zeros(d_model, r2))

        nn.init.kaiming_uniform_(self.P, a=5 ** 0.5)

    def forward(self, shared_out) :
        # [B, T, r1] @ P.T  →  [B, T, r2]
        h = F.linear(shared_out, self.P)         # x @ P.T

        # [B, T, r2] @ B.T  →  [B, T, d_model]
        out = F.linear(h, self.B)                # x @ B.T

        return self.scale * out