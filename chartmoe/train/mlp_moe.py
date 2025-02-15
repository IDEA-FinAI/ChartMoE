"""
    FEATURE: Implementation of MLP-MoE
    AUTHOR: Brian Qu
    URL: https://arxiv.org/abs/2409.03277
    REFERENCE: https://github.com/SHI-Labs/CuMo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack

class MLPMoE(nn.Module):
    def __init__(self, num_experts, num_selected, mm_channels, channels):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.mm_channels = mm_channels
        self.channels = channels

        self.gate = nn.Linear(mm_channels, num_experts, bias=False)
        # nn.init.zeros_(self.gate.weight)

        self.num_selected = num_selected
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(mm_channels, channels, bias=True), 
                    nn.GELU(), 
                    nn.Linear(channels, channels, bias=True)
                )
                for _ in range(num_experts)
            ]
        )
        
    def forward(self, x_img):
        gate_logits = self.gate(x_img)
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x_img.dtype)

        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x_img.dtype)
        
        results = torch.zeros((x_img.shape[0], x_img.shape[1], self.channels)).to(x_img.device, x_img.dtype)
        for b in range(x_img.shape[0]):
            for i, expert in enumerate(self.experts):
                token_idx, nth_expert = torch.where(selected_experts[b] == i)
                results[b][token_idx] += weights[b][token_idx, nth_expert, None] * expert(x_img[b][token_idx])
                
        return results


class MLPMoE_bzloss(nn.Module):
    def __init__(self, num_experts, num_selected, mm_channels, channels):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.mm_channels = mm_channels
        self.channels = channels

        self.gate = nn.Linear(mm_channels, num_experts, bias=False)
        # nn.init.zeros_(self.gate.weight)

        self.num_selected = num_selected
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(mm_channels, channels, bias=True), nn.GELU(), nn.Linear(channels, channels, bias=True)) for _ in range(num_experts)])

    def forward(self, x_img):
        gate_logits = self.gate(x_img)

        router_z_loss = torch.logsumexp(gate_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x_img.dtype)

        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')

        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)

        one_hot_gate_indices = F.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2)

        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x_img.dtype)
        
        results = torch.zeros((x_img.shape[0], x_img.shape[1], self.channels)).to(x_img.device, x_img.dtype)

        for b in range(x_img.shape[0]):
            for i, expert in enumerate(self.experts):
                token_idx, nth_expert = torch.where(selected_experts[b] == i)
                results[b][token_idx] += weights[b][token_idx, nth_expert, None] * expert(x_img[b][token_idx])
        
        return results, 0.1 * balance_loss, 0.01 * router_z_loss