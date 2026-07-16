import torch
import torch.nn as nn
import torch.nn.functional as F


from src.Experts import AlgoExpert, SymmetricExpert, SyntaxPolyExpert, SecureExpert
from src.SharedSubspace import MALoRASharedSubspace
from src.Router import TopKGatingRouter


class MALoRADownProjLayer(nn.Module):

    def __init__(self, original_mlp, n_experts, r1, r2, d_model, d_ffn, alpha):
        super().__init__()

        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.act_fn = original_mlp.act_fn

        self.W_down = original_mlp.down_proj

        for proj in [self.gate_proj, self.up_proj,self.W_down]:
            for p in proj.parameters():
                p.requires_grad = False

        self.last_auxloss = torch.tensor(0.0)
        

        
        self.shared_SA = MALoRASharedSubspace(r1, d_ffn)

        self.router = TopKGatingRouter(d_model, n_experts, k=1)

        self.experts = nn.ModuleList([ AlgoExpert(r1, r2, alpha, d_model),
            SyntaxPolyExpert(r1, r2, alpha, d_model),
            SecureExpert(r1, r2, alpha, d_model),
        ])



    def forward(self, x):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)  # [B,T,d_ffn]

        gate_weights, aux_loss = self.router(x)

        self.last_auxloss = aux_loss

        baseline   = self.W_down(h)  # [B,T,d_model]
        shared_out = self.shared_SA(h)   # [B,T,r1]

        correction = torch.zeros_like(baseline)
        for t, expert in enumerate(self.experts):
            G_t = gate_weights[:, :, t].unsqueeze(-1)
            correction = correction + G_t * expert(shared_out)

        return baseline + correction


class DenseLoRADownProjLayer(nn.Module):
    
    def __init__(self, original_mlp, r, alpha, d_model, d_ffn):
        super().__init__()
        self.gate_proj=original_mlp.gate_proj
        self.up_proj=original_mlp.up_proj
        self.act_fn=original_mlp.act_fn
        self.W_down=original_mlp.down_proj

        for proj in [self.gate_proj,self.up_proj,self.W_down]:
            for p in proj.parameters:
                p.requires_grad=False
        self.scale=alpha/r
        self.A=nn.Parameter(torch.empty(r, d_ffn, dtype=torch.float16))
        self.B=nn.Parameter(torch.zeros(d_model, r, dtype=torch.float16))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.last_auxloss=torch.tensor(0.0)

    
    def forward(self,x):
        h=self.act_fn(self.gate_proj(x))*self.up_proj(x)
        baseline=self.W_down(h)
        correction=F.linear(F.linear(h, self.A), self.B) *self.scale
        return baseline + correction



class SymmetricMoEDownProjLayer(nn.Module):
    def __init__(self, original_mlp, n_experts, r1, r2, d_model, d_ffn, alpha):
        super().__init__()

        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.act_fn = original_mlp.act_fn

        self.W_down = original_mlp.down_proj

        for proj in [self.gate_proj, self.up_proj,self.W_down]:
            for p in proj.parameters():
                p.requires_grad = False

        self.last_auxloss = torch.tensor(0.0)
        

        
        

        self.router = TopKGatingRouter(d_model, n_experts, k=1)

        self.experts = nn.ModuleList([ SymmetricExpert(r1, r2, alpha, d_model, d_ffn),
            SymmetricExpert(r1, r2, alpha, d_model, d_ffn),
            SymmetricExpert(r1, r2, alpha, d_model, d_ffn),
        ])



    def forward(self, x):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)  # [B,T,d_ffn]

        gate_weights, aux_loss = self.router(x)

        self.last_auxloss = aux_loss

        baseline   = self.W_down(h)  # [B,T,d_model]

        correction = torch.zeros_like(baseline)
        for t, expert in enumerate(self.experts):
            G_t = gate_weights[:, :, t].unsqueeze(-1)
            correction = correction + G_t * expert(h)

        return baseline + correction
