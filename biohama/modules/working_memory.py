"""
Working Memory: 동적 메모리 관리
장거리 의존성을 처리하기 위한 메모리 슬롯 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WorkingMemory(nn.Module):
    """
    Working Memory 슬롯 시스템
    정보를 저장하고 검색하는 동적 메모리
    """
    
    def __init__(
        self,
        memory_size: int = 16,
        memory_dim: int = 128,
        key_dim: int = 64,
        value_dim: int = 64
    ):
        """
        Args:
            memory_size: 메모리 슬롯 개수
            memory_dim: 메모리 차원
            key_dim: Key 차원
            value_dim: Value 차원
        """
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # 메모리 슬롯 초기화 (학습 가능)
        self.memory_slots = nn.Parameter(
            torch.randn(memory_size, memory_dim) * 0.1
        )
        
        # Key-Value 프로젝션
        self.key_proj = nn.Linear(memory_dim, key_dim)
        self.value_proj = nn.Linear(memory_dim, value_dim)
        self.query_proj = nn.Linear(memory_dim, key_dim)
        
        # 메모리 업데이트 네트워크
        # value는 memory_dim 크기로 가정 (write 함수에서 value_dim -> memory_dim 변환 후 전달)
        self.update_network = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # 초기화
        self.reset()
    
    def reset(self):
        """메모리 슬롯 초기화"""
        nn.init.normal_(self.memory_slots, mean=0.0, std=0.1)
    
    def write(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        메모리에 정보 쓰기
        
        Args:
            query: 쿼리 텐서 [batch_size, query_dim]
            value: 저장할 값 [batch_size, value_dim]
            memory: 기존 메모리 [batch_size, memory_size, memory_dim] (None이면 self.memory_slots 사용)
            
        Returns:
            업데이트된 메모리 [batch_size, memory_size, memory_dim]
        """
        if memory is None:
            # 배치 차원 확장
            memory = self.memory_slots.unsqueeze(0)  # [1, memory_size, memory_dim]
            batch_size = query.size(0)
            memory = memory.expand(batch_size, -1, -1)
        
        # Key 계산
        keys = self.key_proj(memory)  # [batch_size, memory_size, key_dim]
        query_proj = self.query_proj(query)  # [batch_size, key_dim]
        
        # Attention 가중치 계산 (어떤 슬롯에 쓸지)
        scores = torch.bmm(
            keys,
            query_proj.unsqueeze(-1)  # [batch_size, key_dim, 1]
        ).squeeze(-1)  # [batch_size, memory_size]
        
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, memory_size]
        
        # 메모리 업데이트
        # 각 슬롯에 attention weight에 따라 업데이트
        # value는 memory_dim 크기여야 함 (호출자가 보장)
        assert value.size(-1) == self.memory_dim, \
            f"value dimension {value.size(-1)} must match memory_dim {self.memory_dim}"
        
        value_expanded = value.unsqueeze(1).expand(-1, self.memory_size, -1)
        # [batch_size, memory_size, memory_dim]
        
        # 메모리와 값을 결합하여 업데이트
        combined = torch.cat([memory, value_expanded], dim=-1)
        # [batch_size, memory_size, memory_dim * 2]
        
        update = self.update_network(combined)
        # [batch_size, memory_size, memory_dim]
        
        # Attention weight에 따라 가중 평균
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, memory_size, 1]
        updated_memory = memory + attention_weights * update
        
        return updated_memory
    
    def read(
        self,
        query: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        메모리에서 정보 읽기
        
        Args:
            query: 쿼리 텐서 [batch_size, query_dim]
            memory: 메모리 [batch_size, memory_size, memory_dim] (None이면 self.memory_slots 사용)
            
        Returns:
            retrieved_value: 검색된 값 [batch_size, memory_dim]
            attention_weights: Attention 가중치 [batch_size, memory_size]
        """
        if memory is None:
            memory = self.memory_slots.unsqueeze(0)
            batch_size = query.size(0)
            memory = memory.expand(batch_size, -1, -1)
        
        # Key-Value 계산
        keys = self.key_proj(memory)  # [batch_size, memory_size, key_dim]
        values = self.value_proj(memory)  # [batch_size, memory_size, value_dim]
        query_proj = self.query_proj(query)  # [batch_size, key_dim]
        
        # Attention 계산
        scores = torch.bmm(
            keys,
            query_proj.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, memory_size]
        
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, memory_size]
        
        # 가중 평균으로 값 검색
        retrieved_value = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, memory_size]
            values  # [batch_size, memory_size, value_dim]
        ).squeeze(1)  # [batch_size, value_dim]
        
        # value_dim을 memory_dim으로 프로젝션
        # value_proj는 memory_dim -> value_dim이므로 역변환 필요
        # 간단하게 values를 memory에서 직접 계산
        retrieved_memory = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, memory_size]
            memory  # [batch_size, memory_size, memory_dim]
        ).squeeze(1)  # [batch_size, memory_dim]
        
        return retrieved_memory, attention_weights

