"""
LSTM Baseline 모델
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from biohama.common import BaseModel, TaskSample


class LSTMModel(BaseModel, nn.Module):
    """LSTM Baseline 모델"""
    
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 임베딩 차원
            hidden_dim: Hidden 차원
            num_layers: LSTM 레이어 개수
            dropout: Dropout 비율
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 임베딩
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_seq: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_seq: [batch_size, seq_len]
            task_id: Task ID (사용하지 않음)
            
        Returns:
            출력 로짓 [batch_size, seq_len, vocab_size]
        """
        # 임베딩
        x = self.embedding(input_seq)  # [batch_size, seq_len, embedding_dim]
        x = self.dropout(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        
        # 출력
        logits = self.fc(lstm_out)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def forward_sequence(
        self,
        input_seq: List[int],
        task_id: int
    ) -> List[int]:
        """시퀀스 입력에 대한 예측"""
        self.eval()
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
        
        input_seqs = [s.input for s in batch]
        target_seqs = [s.target for s in batch]
        task_ids = [s.task_id for s in batch]
        
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
        
        logits = self.forward(input_tensor, task_ids[0])
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(
            logits.reshape(-1, self.vocab_size),
            target_tensor.reshape(-1)
        )
        
        predictions = torch.argmax(logits, dim=-1)
        token_acc = (predictions == target_tensor).float().mean().item()
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

