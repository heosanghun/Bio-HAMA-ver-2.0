"""
Transformer Baseline 모델
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Tuple

from biohama.common import BaseModel, TaskSample


class PositionalEncoding(nn.Module):
    """RoPE (Rotary Position Embedding) 스타일의 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(BaseModel, nn.Module):
    """Transformer Baseline 모델 (Decoder-only)"""
    
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_seq: torch.Tensor, task_id: int) -> torch.Tensor:
        # 임베딩 및 위치 인코딩
        x = self.embedding(input_seq)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Causal mask 생성
        seq_len = input_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(input_seq.device)
        
        # Transformer
        x = self.transformer(x, mask=mask)
        
        # 출력
        logits = self.fc(x)
        return logits
    
    def forward_sequence(self, input_seq: List[int], task_id: int) -> List[int]:
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([input_seq], dtype=torch.long)
            logits = self.forward(input_tensor, task_id)
            predictions = torch.argmax(logits, dim=-1)
            return predictions[0].tolist()
    
    def train_step(self, batch: List[TaskSample], device: str = 'cpu') -> Tuple[torch.Tensor, Dict[str, float]]:
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
        loss = loss_fn(logits.reshape(-1, self.vocab_size), target_tensor.reshape(-1))
        
        predictions = torch.argmax(logits, dim=-1)
        token_acc = (predictions == target_tensor).float().mean().item()
        seq_acc = (predictions == target_tensor).all(dim=1).float().mean().item()
        
        metrics = {'loss': loss.item(), 'token_acc': token_acc, 'seq_acc': seq_acc}
        return loss, metrics
    
    def eval_step(self, batch: List[TaskSample], device: str = 'cpu') -> Dict[str, float]:
        return self.train_step(batch, device)

