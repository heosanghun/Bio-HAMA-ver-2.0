"""
Router: 모듈 선택 메커니즘
동적으로 적절한 모듈을 선택하여 정보를 라우팅
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Router(nn.Module):
    """
    Gating 메커니즘을 사용한 모듈 라우터
    입력과 컨텍스트에 따라 적절한 모듈을 선택
    """
    
    def __init__(
        self,
        input_dim: int,
        num_modules: int = 8,
        hidden_dim: int = 128,
        temperature: float = 1.0
    ):
        """
        Args:
            input_dim: 입력 임베딩 차원
            num_modules: 모듈 개수
            hidden_dim: 숨겨진 차원
            temperature: Gating temperature (낮을수록 더 결정적)
        """
        super().__init__()
        self.num_modules = num_modules
        self.temperature = temperature
        
        # Gating 네트워크
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modules)
        )
        
        # Task-specific bias (task_id에 따라 다른 bias)
        self.task_bias = nn.Parameter(torch.zeros(num_modules))
        
    def forward(
        self,
        x: torch.Tensor,
        task_id: int,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력에 대한 모듈 선택 가중치 계산
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, input_dim] 또는 [batch_size, input_dim]
            task_id: Task ID
            return_weights: 가중치 반환 여부
            
        Returns:
            gate_weights: 모듈 선택 가중치 [batch_size, num_modules]
            (선택적으로) raw_logits: 원본 로짓
        """
        # 입력 차원 처리
        if x.dim() == 3:
            # [batch_size, seq_len, input_dim] -> [batch_size, input_dim]
            # 마지막 타임스텝 사용 또는 평균 풀링
            x = x.mean(dim=1)
        elif x.dim() == 2:
            # [batch_size, input_dim]
            pass
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")
        
        # Gating 로짓 계산
        logits = self.gate_network(x) + self.task_bias
        
        # Temperature scaling
        logits = logits / self.temperature
        
        # Softmax로 가중치 계산
        gate_weights = F.softmax(logits, dim=-1)
        
        if return_weights:
            return gate_weights, logits
        return gate_weights
    
    def select_modules(
        self,
        gate_weights: torch.Tensor,
        top_k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-k 모듈 선택
        
        Args:
            gate_weights: 모듈 가중치 [batch_size, num_modules]
            top_k: 선택할 모듈 개수
            
        Returns:
            selected_indices: 선택된 모듈 인덱스 [batch_size, top_k]
            selected_weights: 선택된 가중치 [batch_size, top_k]
        """
        top_k = min(top_k, self.num_modules)
        selected_weights, selected_indices = torch.topk(
            gate_weights, k=top_k, dim=-1
        )
        return selected_indices, selected_weights

