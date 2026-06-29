import torch
import torch.nn as nn
import torch.nn.functional as F

class MALoRASharedSubspace(nn.Module):

    def __init__(self, r1, d_ffn):

        super().__init__()

<<<<<<< HEAD:src/SharedSubspace.py
        self.S_A = nn.Parameter(torch.empty(r1, d_ffn), requires_grad=True)
=======
        self.S_A = nn.Parameter(torch.empty(r1, d_ffn), requires_grad=True
                               )
>>>>>>> f04360202cbb1608f2b66cef82329180a4c57501:SharedSubspace.py

        nn.init.kaiming_uniform_(self.S_A, a=5**0.5)

    def forward(self, x) :
        # torch.matmul(x, self.S_A.T)  →  [B, T, d_ffn] @ [d_ffn, r1]  =  [B, T, r1]

        return F.linear(x, self.S_A) 
