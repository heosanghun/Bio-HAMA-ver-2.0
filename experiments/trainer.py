"""
학습 및 평가 Trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional
from tqdm import tqdm
import wandb

from biohama.common import TaskSample, BaseModel


class TaskDataset(Dataset):
    """Task 샘플 데이터셋"""
    
    def __init__(self, samples: List[TaskSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[TaskSample]) -> List[TaskSample]:
    """TaskSample 배치를 그대로 반환"""
    return batch


class Trainer:
    """모델 학습 및 평가 클래스"""
    
    def __init__(
        self,
        model: BaseModel,
        device: str = 'cpu',
        use_wandb: bool = False,
        project_name: str = 'biohama'
    ):
        """
        Args:
            model: 학습할 모델
            device: 디바이스 ('cpu' or 'cuda')
            use_wandb: WandB 사용 여부
            project_name: WandB 프로젝트 이름
        """
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project=project_name)
            wandb.watch(self.model)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine',
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 10
    ):
        """
        모델 학습
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            num_epochs: 에폭 수
            learning_rate: 학습률
            optimizer_name: 옵티마이저 이름
            scheduler_name: 스케줄러 이름
            gradient_clip: Gradient clipping 값
            early_stopping_patience: Early stopping patience
        """
        # 옵티마이저 설정
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        # 스케줄러 설정
        if scheduler_name.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs
            )
        elif scheduler_name.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        else:
            scheduler = None
        
        best_val_seq_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 학습
            train_metrics = self._train_epoch(train_loader, optimizer, gradient_clip)
            
            # 검증
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._eval_epoch(val_loader)
                
                # Early stopping
                val_seq_acc = val_metrics.get('seq_acc', 0.0)
                if val_seq_acc > best_val_seq_acc:
                    best_val_seq_acc = val_seq_acc
                    patience_counter = 0
                    # 모델 저장
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # 스케줄러 업데이트
            if scheduler is not None:
                scheduler.step()
            
            # 로깅
            metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            metrics['epoch'] = epoch + 1
            metrics['lr'] = optimizer.param_groups[0]['lr']
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # 콘솔 출력
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Token Acc: {train_metrics['token_acc']:.4f}, "
                  f"Seq Acc: {train_metrics['seq_acc']:.4f}")
            if val_metrics:
                print(f"  Val - Loss: {val_metrics['loss']:.4f}, "
                      f"Token Acc: {val_metrics['token_acc']:.4f}, "
                      f"Seq Acc: {val_metrics['seq_acc']:.4f}")
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        gradient_clip: float
    ) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        total_token_acc = 0.0
        total_seq_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # 배치를 리스트로 변환
            if isinstance(batch, list):
                batch_list = batch
            else:
                batch_list = [batch]
            
            # 학습 스텝
            loss, metrics = self.model.train_step(batch_list, self.device)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip
                )
            
            optimizer.step()
            
            # 메트릭 누적
            total_loss += metrics['loss']
            total_token_acc += metrics['token_acc']
            total_seq_acc += metrics['seq_acc']
            num_batches += 1
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'seq_acc': f"{metrics['seq_acc']:.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'token_acc': total_token_acc / num_batches,
            'seq_acc': total_seq_acc / num_batches
        }
    
    def _eval_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """한 에폭 평가"""
        self.model.eval()
        total_loss = 0.0
        total_token_acc = 0.0
        total_seq_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                if isinstance(batch, list):
                    batch_list = batch
                else:
                    batch_list = [batch]
                
                _, metrics = self.model.eval_step(batch_list, self.device)
                
                total_loss += metrics['loss']
                total_token_acc += metrics['token_acc']
                total_seq_acc += metrics['seq_acc']
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'seq_acc': f"{metrics['seq_acc']:.4f}"
                })
        
        return {
            'loss': total_loss / num_batches,
            'token_acc': total_token_acc / num_batches,
            'seq_acc': total_seq_acc / num_batches
        }
    
    def evaluate(
        self,
        test_loaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """
        여러 테스트 세트 평가
        
        Args:
            test_loaders: {'split_name': DataLoader} 형태의 딕셔너리
            
        Returns:
            {'split_name': {'metric': value}} 형태의 결과
        """
        results = {}
        
        for split_name, loader in test_loaders.items():
            print(f"\nEvaluating on {split_name}...")
            metrics = self._eval_epoch(loader)
            results[split_name] = metrics
            
            print(f"{split_name} Results:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Token Accuracy: {metrics['token_acc']:.4f}")
            print(f"  Sequence Accuracy: {metrics['seq_acc']:.4f}")
        
        return results

