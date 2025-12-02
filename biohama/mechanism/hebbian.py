"""
Hebbian Learning: 모듈 간 연결 학습
"Neurons that fire together, wire together" 원칙
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class HebbianLearning(nn.Module):
    """
    Hebbian Learning 메커니즘
    모듈 간 활성화 패턴을 학습하여 연결 강도 업데이트
    """
    
    def __init__(
        self,
        num_modules: int = 8,
        connection_dim: int = 128,
        learning_rate: float = 0.01,
        decay_rate: float = 0.95
    ):
        """
        Args:
            num_modules: 모듈 개수
            connection_dim: 연결 차원
            learning_rate: Hebbian 학습률
            decay_rate: 연결 강도 감쇠율
        """
        super().__init__()
        self.num_modules = num_modules
        self.connection_dim = connection_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # 모듈 간 연결 강도 행렬 (대칭)
        # connection_matrix[i, j] = 모듈 i와 j 간의 연결 강도
        self.register_buffer(
            'connection_matrix',
            torch.eye(num_modules) * 0.1  # 초기에는 약한 자기 연결만
        )
        
        # 모듈 활성화 히스토리 (최근 N 스텝)
        self.max_history = 10
        self.activation_history = []
    
    def update_connections(
        self,
        module_states: torch.Tensor,
        module_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        모듈 활성화에 기반하여 연결 강도 업데이트
        
        Args:
            module_states: 모듈 상태 [batch_size, num_modules, state_dim]
            module_weights: 모듈 선택 가중치 [batch_size, num_modules]
            
        Returns:
            업데이트된 연결 강도 행렬
        """
        batch_size = module_states.size(0)
        
        # 활성화된 모듈 쌍 찾기
        # module_weights를 사용하여 동시에 활성화된 모듈 쌍 강화
        coactivation = torch.bmm(
            module_weights.unsqueeze(-1),  # [batch_size, num_modules, 1]
            module_weights.unsqueeze(1)    # [batch_size, 1, num_modules]
        )  # [batch_size, num_modules, num_modules]
        
        # 배치 평균
        coactivation = coactivation.mean(dim=0)  # [num_modules, num_modules]
        
        # Hebbian 규칙: 동시 활성화 시 연결 강화
        # ΔW_ij = η * (A_i * A_j - decay * W_ij)
        hebbian_update = self.learning_rate * (
            coactivation - self.decay_rate * self.connection_matrix
        )
        
        # 대칭 행렬 유지
        hebbian_update = (hebbian_update + hebbian_update.T) / 2
        
        # 연결 강도 업데이트
        self.connection_matrix = torch.clamp(
            self.connection_matrix + hebbian_update,
            min=0.0,  # 연결 강도는 음수가 될 수 없음
            max=1.0   # 최대값 제한
        )
        
        return self.connection_matrix
    
    def get_connection_strength(
        self,
        module_i: int,
        module_j: int
    ) -> float:
        """모듈 i와 j 간의 연결 강도 반환"""
        return self.connection_matrix[module_i, module_j].item()
    
    def get_connected_modules(
        self,
        module_id: int,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        특정 모듈과 강하게 연결된 모듈들 반환
        
        Args:
            module_id: 기준 모듈 ID
            threshold: 연결 강도 임계값
            
        Returns:
            연결된 모듈 인덱스
        """
        connections = self.connection_matrix[module_id]
        return torch.where(connections > threshold)[0]
    
    def reset(self):
        """연결 강도 초기화"""
        self.connection_matrix.fill_(0.0)
        self.connection_matrix.fill_diagonal_(0.1)
        self.activation_history = []

