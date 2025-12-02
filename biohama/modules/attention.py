"""
Sparse Attention: 희소 주의 메커니즘
효율적인 장거리 의존성 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SparseAttention(nn.Module):
    """
    Sparse Attention 메커니즘
    전체 시퀀스가 아닌 일부 토큰에만 attention을 적용
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        sparsity: float = 0.5,
        dropout: float = 0.1
    ):
        """
        Args:
            dim: 임베딩 차원
            num_heads: Attention head 개수
            sparsity: 희소도 (0.0 ~ 1.0, 높을수록 더 희소)
            dropout: Dropout 비율
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity = sparsity
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V 프로젝션
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse Attention 계산
        
        Args:
            query: [batch_size, seq_len_q, dim]
            key: [batch_size, seq_len_k, dim]
            value: [batch_size, seq_len_k, dim]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            output: [batch_size, seq_len_q, dim]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Q, K, V 프로젝션 및 head 분할
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Sparse masking: 상위 (1-sparsity) 비율만 유지
        if self.sparsity > 0 and self.training:
            # 각 query에 대해 top-k key만 선택
            k = max(1, int(seq_len_k * (1 - self.sparsity)))
            topk_values, topk_indices = torch.topk(scores, k=k, dim=-1)
            
            # Sparse scores 생성
            sparse_scores = torch.full_like(scores, float('-inf'))
            sparse_scores.scatter_(-1, topk_indices, topk_values)
            scores = sparse_scores
        
        # Mask 적용
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Value와 곱하기
        output = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, seq_len_q, head_dim]
        
        # Head 결합
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.dim
        )
        
        # 출력 프로젝션
        output = self.out_proj(output)
        
        return output, attention_weights

