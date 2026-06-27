import torch
import torch.nn as nn
import torch.nn.functional as F


from Experts import AlgoExpert, SyntaxPolyExpert, SecureExpert
from SharedSubspace import MALoRASharedSubspace
from Router import TopKGatingRouter


class MALoRADownProjLayer(nn.Module):

    def __init__(self, original_mlp, n_experts, r1, r2, d_model, d_ffn, alpha):
        super().__init__()

       
        # self.W_down = nn.Linear(d_ffn, d_model, bias=False)

        # self.W_down.weight = nn.Parameter(W_down_weight.clone(), requires_grad=False)



        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.act_fn = original_mlp.act_fn

        self.W_down = nn.Linear(d_ffn, d_model, bias=False)


        with torch.no_grad():
            self.W_down.weight = nn.Parameter(original_mlp.down_proj.weight.clone() , requires_grad=False)
        #     self.W_down.weight.copy_(original_mlp.down_proj.weight)
        # self.W_down.weight.requires_grad = False

        for proj in [self.gate_proj, self.up_proj]:
            for p in proj.parameters():
                p.requires_grad = False

        
        self.shared_SA = MALoRASharedSubspace(r1=r1, d_ffn=d_ffn)

        
        self.router = TopKGatingRouter(d_model=d_model, n_experts=n_experts, k=1)


        self.experts = nn.ModuleList([ AlgoExpert(r1, r2, alpha, d_model),
            SyntaxPolyExpert(r1, r2, alpha, d_model),
            SecureExpert(r1, r2, alpha, d_model),
        ])



    def forward(self, x, gate_weights):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)  # [B,T,d_ffn]

        gate_weights, aux_loss = self.router(x)

        baseline   = self.W_down(h)       # [B,T,d_model]
        shared_out = self.shared_SA(h)    # [B,T,r1]

        correction = torch.zeros_like(baseline)
        for t, expert in enumerate(self.experts):
            G_t = gate_weights[:, :, t].unsqueeze(-1)
            correction = correction + G_t * expert(shared_out)

        return baseline + correction