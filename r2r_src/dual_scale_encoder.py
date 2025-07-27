import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarseScaleEncoder(nn.Module):
    def __init__(self, node_dim, text_dim, hidden_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True, dim_feedforward=hidden_dim*2, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ffn = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, text_features, adj_matrix):
        """
        node_features: [num_nodes, node_dim]
        text_features: [text_dim]
        adj_matrix: [num_nodes, num_nodes] 
        """
        num_nodes = node_features.size(0)
        node_emb = self.node_proj(node_features)  # [num_nodes, hidden_dim]
        text_emb = self.text_proj(text_features).unsqueeze(0)  # [1, hidden_dim]
        fused = node_emb + text_emb  # [num_nodes, hidden_dim]

        # Construct attention mask, only allow attention between neighbor nodes
        # mask: [num_nodes, num_nodes], 0=visible, -inf=invisible
        attn_mask = (adj_matrix == 0).float() * -1e9
        # Self is always visible to itself
        attn_mask.fill_diagonal_(0)

        # transformer requires batch dimension
        fused = fused.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        out = self.transformer(fused, mask=attn_mask)  # [1, num_nodes, hidden_dim]
        out = out.squeeze(0)  # [num_nodes, hidden_dim]

        scores = self.ffn(out).squeeze(-1)  # [num_nodes]
        return scores, out

class FineScaleEncoder(nn.Module):
    def __init__(self, pano_dim, text_dim, hidden_dim, num_heads=2):
        super().__init__()
        self.pano_proj = nn.Linear(pano_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Linear(hidden_dim, 1)

    def forward(self, pano_features, text_features):
        """
        pano_features: [num_panos, pano_dim]
        text_features: [text_dim]
        """
        pano_emb = self.pano_proj(pano_features)  # [num_panos, hidden_dim]
        text_emb = self.text_proj(text_features).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

        # cross-attention: query=text, key/value=pano
        # output: attn_out [1, 1, hidden_dim], attn_weights [1, 1, num_panos]
        attn_out, attn_weights = self.cross_attn(text_emb, pano_emb.unsqueeze(0), pano_emb.unsqueeze(0))
        # Copy attn_out to each pano (or directly use pano_emb + attn_out)
        attn_out = attn_out.repeat(1, pano_emb.size(0), 1).squeeze(0)  # [num_panos, hidden_dim]
        fused = pano_emb + attn_out  # [num_panos, hidden_dim]
        scores = self.ffn(fused).squeeze(-1)  # [num_panos]
        return scores, fused

class DynamicFusion(nn.Module):
    def __init__(self, global_dim, local_dim):
        super().__init__()
        self.fusion_ffn = nn.Sequential(
            nn.Linear(global_dim + local_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, global_emb, local_emb, global_scores, local_scores, candidate_indices):
        """
        global_emb: [num_candidates, D]
        local_emb: [num_candidates, D]
        global_scores: [num_nodes]
        local_scores: [num_candidates]
        candidate_indices: [num_candidates]
        """
        fused_scores = global_scores.clone()
        # Only fuse on candidate nodes
        fusion_input = torch.cat([global_emb, local_emb], dim=-1)  # [num_candidates, 2D]
        sigma = self.fusion_ffn(fusion_input).squeeze(-1)          # [num_candidates]
        fused_scores_duet = sigma * global_scores[candidate_indices] + (1 - sigma) * local_scores
        fused_scores[candidate_indices] = fused_scores_duet
        return fused_scores, sigma
