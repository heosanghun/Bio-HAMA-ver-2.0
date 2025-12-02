"""
BioHama 실험 메인 실행 스크립트
"""

import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from experiments.trainer import Trainer, TaskDataset, collate_fn
from experiments.tasks.generators import (
    CopyTask, ReverseTask, SortTask, DelayedAssociativeRecallTask,
    generate_dataset
)


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


def instantiate_model(cfg: DictConfig):
    """모델 인스턴스 생성"""
    model_cfg = cfg.model
    model_class_path = model_cfg._model_
    
    # 모듈 경로에서 클래스 가져오기
    module_path, class_name = model_class_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    # 모델 생성 (모델 설정에서 _model_ 제외)
    model_params = {k: v for k, v in model_cfg.items() if k != '_model_'}
    model = model_class(**model_params)
    
    return model


def instantiate_task_generator(cfg: DictConfig):
    """Task Generator 인스턴스 생성"""
    task_cfg = cfg.task
    task_class_path = task_cfg._task_
    
    module_path, class_name = task_class_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    task_class = getattr(module, class_name)
    
    task_params = {k: v for k, v in task_cfg.items() if k != '_task_'}
    task_generator = task_class(**task_params)
    
    return task_generator


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """메인 실행 함수"""
    print("=" * 60)
    print("BioHama PoC 실험 시작")
    print("=" * 60)
    print(f"\n설정:\n{OmegaConf.to_yaml(cfg)}")
    
    # 시드 설정
    set_seed(cfg.experiment.seed)
    
    # 디바이스 설정
    device = cfg.experiment.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        device = 'cpu'
    
    # 모델 생성
    print("\n모델 생성 중...")
    model = instantiate_model(cfg)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Task Generator 생성
    print("\nTask Generator 생성 중...")
    task_generator = instantiate_task_generator(cfg)
    
    # 데이터셋 생성
    print("\n데이터셋 생성 중...")
    data_cfg = cfg.data
    
    train_samples = generate_dataset(
        task_generator, data_cfg.train_size, split="train"
    )
    id_test_samples = generate_dataset(
        task_generator, data_cfg.id_test_size, split="id_test"
    )
    ood_test_samples = generate_dataset(
        task_generator, data_cfg.ood_test_size, split="ood_test"
    )
    hard_test_samples = generate_dataset(
        task_generator, data_cfg.hard_test_size, split="hard"
    )
    
    print(f"  Train: {len(train_samples)} samples")
    print(f"  ID Test: {len(id_test_samples)} samples")
    print(f"  OOD Test: {len(ood_test_samples)} samples")
    print(f"  HARD Test: {len(hard_test_samples)} samples")
    
    # DataLoader 생성
    train_loader = DataLoader(
        TaskDataset(train_samples),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    id_test_loader = DataLoader(
        TaskDataset(id_test_samples),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    ood_test_loader = DataLoader(
        TaskDataset(ood_test_samples),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    hard_test_loader = DataLoader(
        TaskDataset(hard_test_samples),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        device=device,
        use_wandb=cfg.logging.use_wandb,
        project_name=cfg.logging.wandb_project
    )
    
    # 학습
    print("\n학습 시작...")
    trainer.train(
        train_loader=train_loader,
        val_loader=id_test_loader,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        optimizer_name=cfg.training.optimizer,
        scheduler_name=cfg.training.scheduler,
        gradient_clip=cfg.training.gradient_clip,
        early_stopping_patience=cfg.training.early_stopping_patience
    )
    
    # 최고 모델 로드
    if os.path.exists('best_model.pt'):
        print("\n최고 모델 로드 중...")
        model.load_state_dict(torch.load('best_model.pt'))
    
    # 평가
    print("\n" + "=" * 60)
    print("최종 평가")
    print("=" * 60)
    
    test_loaders = {
        'id_test': id_test_loader,
        'ood_test': ood_test_loader,
        'hard_test': hard_test_loader
    }
    
    results = trainer.evaluate(test_loaders)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"\n{'Split':<15} {'Loss':<10} {'Token Acc':<12} {'Seq Acc':<12}")
    print("-" * 60)
    for split_name, metrics in results.items():
        print(f"{split_name:<15} {metrics['loss']:<10.4f} "
              f"{metrics['token_acc']:<12.4f} {metrics['seq_acc']:<12.4f}")
    
    # 성공 조건 확인
    print("\n" + "=" * 60)
    print("성공 조건 확인")
    print("=" * 60)
    
    train_result = trainer._eval_epoch(train_loader)
    id_result = results['id_test']
    ood_result = results['ood_test']
    hard_result = results['hard_test']
    
    success_criteria = {
        'Train Seq-Acc > 95%': train_result['seq_acc'] > 0.95,
        'ID Test Seq-Acc > 90%': id_result['seq_acc'] > 0.90,
        'OOD Test Seq-Acc > 70%': ood_result['seq_acc'] > 0.70,
        'HARD Test 성능': hard_result['seq_acc'] > 0.50
    }
    
    for criterion, passed in success_criteria.items():
        status = "✓" if passed else "✗"
        print(f"{status} {criterion}")
    
    print("\n실험 완료!")


if __name__ == "__main__":
    main()

