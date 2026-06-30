import torch

import torch.nn as nn

class TopKGatingRouter(nn.Module):
    
    def __init__(self,d_model,n_experts,k,alpha=0.01):
        super().__init__()
        self.n_experts=n_experts
        self.k=k
        self.d_model=d_model
        self.Wg=nn.Linear(d_model,n_experts,bias=False)
        self.Wg = self.Wg.to(torch.float16)
        self.alpha=alpha



    def _load_balance_loss(self,weights,scores):
        chosen_expert=torch.argmax(weights,dim=-1)#shape=[4,128]
        total_tokens=chosen_expert.numel()
        f = scores.new_zeros(self.n_experts)
        for i in range(self.n_experts):
           tokens_for_expert=(chosen_expert==i).sum()
           f[i]=tokens_for_expert.float()/total_tokens
        scores_softmax=torch.softmax(scores,dim=-1)
        p=torch.sum(scores_softmax,dim=(0,1))/total_tokens
        aux_loss=self.alpha*self.n_experts*torch.dot(f,p)
        return aux_loss
    



    def forward(self,x):
        scores=self.Wg(x)
        if torch.isnan(scores).any():
            raise RuntimeError("NaN in scores")
        if self.training:
            noise=torch.randn_like(scores)*0.01
            scores+=noise
        values,indices=torch.topk(scores,self.k,dim=-1)
        container=torch.full_like(scores, float('-inf'))
        container.scatter_(-1,indices,values)
        routing_weights=torch.softmax(container,dim=-1)
        if torch.isnan(routing_weights).any():
            raise RuntimeError("NaN in routing_weights")
        aux_loss=self._load_balance_loss(routing_weights,scores)
        if torch.isnan(aux_loss):
            raise RuntimeError("NaN in aux_loss")
        return routing_weights,aux_loss
    


    
                
                
