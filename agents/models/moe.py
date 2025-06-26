import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, gate_dim, hidden_dim, top_k, dropout=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim))
            for _ in range(num_experts)])
        self.gate = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, gate_dim), nn.ReLU(inplace=True),\
            nn.Dropout(dropout), nn.Linear(gate_dim, num_experts))])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, x, metrics=None):
        batch_size = x.size(0)

        # Gate scores
        gate_scores_logits_ = self.gate[0](x)
        gate_scores_logits = gate_scores_logits_
        gate_scores = F.softmax(gate_scores_logits, dim=1)

        # Top-k gate scores and indices
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=1)

        if metrics is not None:
            for i in range(self.num_experts):
                metrics['expert_{}_usage_rate'.format(i)] = (top_k_indices == i).sum().item() / batch_size

        # Pass inputs through all experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        expert_outputs = expert_outputs.permute(1, 0, 2)

        # Advanced indexing for selecting top-k expert outputs
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k).reshape(-1).to(x.device)
        selected_expert_outputs = expert_outputs[batch_indices, top_k_indices.reshape(-1)]
        selected_expert_outputs = selected_expert_outputs.reshape(batch_size, self.top_k, self.output_dim)

        # Scale the selected expert outputs by the corresponding gate scores
        scaled_expert_outputs = selected_expert_outputs * top_k_scores.unsqueeze(2)
        scaled_expert_outputs /= (top_k_scores.sum(dim=1, keepdim=True).unsqueeze(2) + 1e-9)

        # Sum the scaled expert outputs for the final output
        combined_output = scaled_expert_outputs.sum(dim=1)

        aux_loss = self.moe_auxiliary_loss(gate_scores, top_k_indices)

        return combined_output, aux_loss

    def moe_auxiliary_loss(self, gate_scores, top_k_indices, lambda_balance=1.0, lambda_entropy=1.0):
        batch_size, num_experts = gate_scores.size()

        # Load Balancing Loss
        one_hot = F.one_hot(top_k_indices, num_classes=num_experts).float()
        expert_load = one_hot.sum(dim=[0, 1]) / (batch_size + 1e-9)
        load_balancing_loss = expert_load.var()

        # Entropy Loss
        entropy = -(gate_scores * torch.log(gate_scores + 1e-9)).sum(dim=1).mean()

        # Combine the losses
        auxiliary_loss = lambda_balance * load_balancing_loss + lambda_entropy * entropy

        return auxiliary_loss