"""
간단한 실험 실행 스크립트 (Hydra 없이)
"""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.trainer import Trainer, TaskDataset, collate_fn
from experiments.tasks.generators import CopyTask, generate_dataset
from biohama.biohama_model import BioHamaModel


def set_seed(seed: int):
    """랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """간단한 실험 실행"""
    print("=" * 60)
    print("BioHama PoC 간단 실험")
    print("=" * 60)
    
    # 시드 설정
    set_seed(42)
    
    # 디바이스
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 모델 생성
    print("\n모델 생성 중...")
    model = BioHamaModel(
        vocab_size=100,
        embedding_dim=128,
        num_modules=8,
        memory_size=16,
        num_heads=4,
        max_seq_len=100,
        dropout=0.1
    )
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Task Generator
    task_generator = CopyTask(vocab_size=100, min_length=5, max_length=20)
    
    # 데이터셋 생성 (작은 크기로 빠른 테스트)
    print("\n데이터셋 생성 중...")
    train_samples = generate_dataset(task_generator, num_samples=1000, split="train")
    id_test_samples = generate_dataset(task_generator, num_samples=100, split="id_test")
    ood_test_samples = generate_dataset(task_generator, num_samples=100, split="ood_test")
    
    print(f"  Train: {len(train_samples)} samples")
    print(f"  ID Test: {len(id_test_samples)} samples")
    print(f"  OOD Test: {len(ood_test_samples)} samples")
    
    # DataLoader
    train_loader = DataLoader(
        TaskDataset(train_samples),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    id_test_loader = DataLoader(
        TaskDataset(id_test_samples),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    ood_test_loader = DataLoader(
        TaskDataset(ood_test_samples),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        use_wandb=False
    )
    
    # 학습 (짧은 에폭으로 빠른 테스트)
    print("\n학습 시작...")
    trainer.train(
        train_loader=train_loader,
        val_loader=id_test_loader,
        num_epochs=10,
        learning_rate=1e-4,
        optimizer_name='adamw',
        scheduler_name='cosine',
        gradient_clip=1.0,
        early_stopping_patience=5
    )
    
    # 평가
    print("\n" + "=" * 60)
    print("최종 평가")
    print("=" * 60)
    
    test_loaders = {
        'id_test': id_test_loader,
        'ood_test': ood_test_loader
    }
    
    results = trainer.evaluate(test_loaders)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"\n{'Split':<15} {'Loss':<10} {'Token Acc':<12} {'Seq Acc':<12}")
    print("-" * 60)
    for split_name, metrics in results.items():
        print(f"{split_name:<15} {metrics['loss']:<10.4f} "
              f"{metrics['token_acc']:<12.4f} {metrics['seq_acc']:<12.4f}")
    
    print("\n실험 완료!")


if __name__ == "__main__":
    main()

