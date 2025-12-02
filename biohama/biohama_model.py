"""
BioHama 통합 모델
모든 구성요소를 통합한 메인 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

from .modules.router import Router
from .modules.working_memory import WorkingMemory
from .modules.attention import SparseAttention
from .mechanism.hebbian import HebbianLearning
from .mechanism.message_passing import MessagePassing
from .common import BaseModel, TaskSample


class Module(nn.Module):
    """개별 처리 모듈"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 모듈 내부 처리 네트워크
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """모듈 forward pass"""
        return self.network(x)


class BioHamaModel(BaseModel, nn.Module):
    """
    BioHama 통합 모델
    Router, Working Memory, Hebbian Learning, Message Passing을 통합
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 128,
        num_modules: int = 8,
        memory_size: int = 16,
        num_heads: int = 4,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 임베딩 차원
            num_modules: 모듈 개수
            memory_size: Working Memory 슬롯 개수
            num_heads: Attention head 개수
            max_seq_len: 최대 시퀀스 길이
            dropout: Dropout 비율
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_modules = num_modules
        self.memory_size = memory_size
        self.max_seq_len = max_seq_len
        
        # 임베딩
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, embedding_dim) * 0.1
        )
        
        # 핵심 구성요소
        self.router = Router(
            input_dim=embedding_dim,
            num_modules=num_modules,
            hidden_dim=embedding_dim
        )
        
        self.working_memory = WorkingMemory(
            memory_size=memory_size,
            memory_dim=embedding_dim
        )
        
        self.attention = SparseAttention(
            dim=embedding_dim,
            num_heads=num_heads,
            sparsity=0.3,
            dropout=dropout
        )
        
        self.hebbian = HebbianLearning(
            num_modules=num_modules,
            connection_dim=embedding_dim
        )
        
        self.message_passing = MessagePassing(
            num_modules=num_modules,
            message_dim=embedding_dim,
            num_steps=2
        )
        
        # 처리 모듈들
        self.modules_list = nn.ModuleList([
            Module(embedding_dim, embedding_dim, embedding_dim)
            for _ in range(num_modules)
        ])
        
        # 출력 디코더
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Working Memory 상태 저장
        self.current_memory = None
    
    def reset_memory(self):
        """Working Memory 초기화"""
        self.working_memory.reset()
        self.current_memory = None
    
    def forward(
        self,
        input_seq: torch.Tensor,
        task_id: int,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_seq: 입력 시퀀스 [batch_size, seq_len]
            task_id: Task ID
            return_attention: Attention 가중치 반환 여부
            
        Returns:
            출력 로짓 [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_seq.shape
        
        # 임베딩 및 위치 인코딩
        x = self.embedding(input_seq)  # [batch_size, seq_len, embedding_dim]
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Working Memory 초기화 (새 시퀀스 시작 시)
        if self.current_memory is None:
            self.current_memory = self.working_memory.memory_slots.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        
        # 시퀀스를 순차적으로 처리
        outputs = []
        module_states_list = []
        
        for t in range(seq_len):
            # 현재 타임스텝의 입력
            current_input = x[:, t, :]  # [batch_size, embedding_dim]
            
            # Router로 모듈 선택
            gate_weights = self.router(current_input, task_id)
            # [batch_size, num_modules]
            
            # Top-k 모듈 선택
            selected_indices, selected_weights = self.router.select_modules(
                gate_weights, top_k=2
            )
            
            # Working Memory에서 정보 읽기
            retrieved_value, mem_attention = self.working_memory.read(
                current_input, self.current_memory
            )
            
            # 입력과 메모리 정보 결합
            combined_input = current_input + retrieved_value
            combined_input = self.layer_norm(combined_input)
            
            # 선택된 모듈들로 처리
            module_outputs = []
            for i in range(self.num_modules):
                module_output = self.modules_list[i](combined_input)
                # 모듈 가중치 적용
                weight = gate_weights[:, i:i+1]  # [batch_size, 1]
                weighted_output = module_output * weight
                module_outputs.append(weighted_output)
            
            # 모듈 출력 집계
            aggregated_output = sum(module_outputs)  # [batch_size, embedding_dim]
            
            # Message Passing (모듈 간 정보 전달)
            module_states = torch.stack([
                self.modules_list[i](combined_input)
                for i in range(self.num_modules)
            ], dim=1)  # [batch_size, num_modules, embedding_dim]
            
            updated_states = self.message_passing(
                module_states,
                self.hebbian.connection_matrix,
                gate_weights
            )
            
            # 집계된 출력에 message passing 결과 반영
            message_output = updated_states.mean(dim=1)  # [batch_size, embedding_dim]
            aggregated_output = aggregated_output + message_output
            
            # Working Memory에 정보 쓰기
            self.current_memory = self.working_memory.write(
                aggregated_output,
                aggregated_output,
                self.current_memory
            )
            
            # Hebbian Learning 업데이트
            self.hebbian.update_connections(
                updated_states,  # [batch_size, num_modules, embedding_dim]
                gate_weights
            )
            
            outputs.append(aggregated_output)
            module_states_list.append(module_states)
        
        # 출력 시퀀스 구성
        output_seq = torch.stack(outputs, dim=1)  # [batch_size, seq_len, embedding_dim]
        
        # Sparse Attention 적용
        output_seq, attention_weights = self.attention(
            output_seq, output_seq, output_seq
        )
        
        # 디코딩
        logits = self.decoder(output_seq)  # [batch_size, seq_len, vocab_size]
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def forward_sequence(
        self,
        input_seq: List[int],
        task_id: int
    ) -> List[int]:
        """
        시퀀스 입력에 대한 예측
        
        Args:
            input_seq: 입력 시퀀스 (Token IDs)
            task_id: Task ID
            
        Returns:
            예측된 출력 시퀀스
        """
        self.eval()
        self.reset_memory()
        
        with torch.no_grad():
            input_tensor = torch.tensor([input_seq], dtype=torch.long)
            logits = self.forward(input_tensor, task_id)
            predictions = torch.argmax(logits, dim=-1)
            return predictions[0].tolist()
    
    def train_step(
        self,
        batch: List[TaskSample],
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """학습 스텝"""
        self.train()
        self.reset_memory()
        
        # 배치 구성
        input_seqs = []
        target_seqs = []
        task_ids = []
        
        for sample in batch:
            input_seqs.append(sample.input)
            target_seqs.append(sample.target)
            task_ids.append(sample.task_id)
        
        # 패딩
        max_len = max(len(seq) for seq in input_seqs + target_seqs)
        
        def pad_sequence(seq, max_len):
            return seq + [0] * (max_len - len(seq))
        
        input_tensor = torch.tensor(
            [pad_sequence(seq, max_len) for seq in input_seqs],
            dtype=torch.long
        ).to(device)
        
        target_tensor = torch.tensor(
            [pad_sequence(seq, max_len) for seq in target_seqs],
            dtype=torch.long
        ).to(device)
        
        # Forward
        logits = self.forward(input_tensor, task_ids[0])
        
        # Loss 계산
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(
            logits.reshape(-1, self.vocab_size),
            target_tensor.reshape(-1)
        )
        
        # Accuracy 계산
        predictions = torch.argmax(logits, dim=-1)
        token_acc = (predictions == target_tensor).float().mean().item()
        
        # Sequence accuracy
        seq_acc = (predictions == target_tensor).all(dim=1).float().mean().item()
        
        metrics = {
            'loss': loss.item(),
            'token_acc': token_acc,
            'seq_acc': seq_acc
        }
        
        return loss, metrics
    
    def eval_step(
        self,
        batch: List[TaskSample],
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """평가 스텝"""
        return self.train_step(batch, device)

