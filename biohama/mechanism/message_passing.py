"""
Message Passing: 모듈 간 정보 전달
그래프 기반 메시지 전달 메커니즘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MessagePassing(nn.Module):
    """
    모듈 간 메시지 전달 메커니즘
    Hebbian 연결 강도를 기반으로 정보를 전달
    """
    
    def __init__(
        self,
        num_modules: int = 8,
        message_dim: int = 128,
        num_steps: int = 2
    ):
        """
        Args:
            num_modules: 모듈 개수
            message_dim: 메시지 차원
            num_steps: 메시지 전달 스텝 수
        """
        super().__init__()
        self.num_modules = num_modules
        self.message_dim = message_dim
        self.num_steps = num_steps
        
        # 메시지 생성 네트워크
        self.message_network = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # 메시지 집계 네트워크
        self.aggregate_network = nn.Sequential(
            nn.Linear(message_dim * 2, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
    
    def forward(
        self,
        module_states: torch.Tensor,
        connection_matrix: torch.Tensor,
        module_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        모듈 간 메시지 전달
        
        Args:
            module_states: 모듈 상태 [batch_size, num_modules, message_dim]
            connection_matrix: 연결 강도 행렬 [num_modules, num_modules]
            module_weights: 모듈 가중치 [batch_size, num_modules] (선택적)
            
        Returns:
            업데이트된 모듈 상태 [batch_size, num_modules, message_dim]
        """
        batch_size = module_states.size(0)
        current_states = module_states
        
        # 여러 스텝에 걸쳐 메시지 전달
        for step in range(self.num_steps):
            # 각 모듈에서 메시지 생성
            messages = self.message_network(current_states)
            # [batch_size, num_modules, message_dim]
            
            # 연결 강도에 따라 메시지 전달
            # connection_matrix를 사용하여 가중 평균
            connection_weights = connection_matrix.unsqueeze(0)  # [1, num_modules, num_modules]
            
            # 각 모듈이 받는 메시지 계산
            received_messages = torch.bmm(
                connection_weights.expand(batch_size, -1, -1),
                messages
            )  # [batch_size, num_modules, message_dim]
            
            # 모듈 가중치 적용 (활성화된 모듈만 메시지 전달)
            if module_weights is not None:
                module_weights_expanded = module_weights.unsqueeze(-1)  # [batch_size, num_modules, 1]
                received_messages = received_messages * module_weights_expanded
            
            # 현재 상태와 받은 메시지 결합
            combined = torch.cat([current_states, received_messages], dim=-1)
            # [batch_size, num_modules, message_dim * 2]
            
            # 집계 및 업데이트
            updated_states = self.aggregate_network(combined)
            # [batch_size, num_modules, message_dim]
            
            # Residual connection
            current_states = current_states + updated_states
        
        return current_states
    
    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        connection_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        연결 강도를 고려한 Attention 계산
        
        Args:
            query: [batch_size, num_modules, dim]
            key: [batch_size, num_modules, dim]
            connection_matrix: [num_modules, num_modules]
            
        Returns:
            Attention 가중치 [batch_size, num_modules, num_modules]
        """
        # 표준 Attention 계산
        scores = torch.bmm(query, key.transpose(-2, -1))
        # [batch_size, num_modules, num_modules]
        
        # 연결 강도로 조정
        connection_bias = connection_matrix.unsqueeze(0)  # [1, num_modules, num_modules]
        scores = scores + connection_bias
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights

