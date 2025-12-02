"""
GRU Baseline 모델
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from biohama.common import BaseModel, TaskSample


class GRUModel(BaseModel, nn.Module):
    """GRU Baseline 모델"""
    
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_seq: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.embedding(input_seq)
        x = self.dropout(x)
        gru_out, _ = self.gru(x)
        logits = self.fc(gru_out)
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

