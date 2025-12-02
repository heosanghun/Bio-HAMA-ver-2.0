"""
Task Generator: 다양한 알고리즘 추론 태스크 생성
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from biohama.common import TaskSample


class TaskGenerator:
    """Task Generator 기본 클래스"""
    
    def __init__(
        self,
        vocab_size: int = 100,
        min_length: int = 3,
        max_length: int = 10,
        noise_level: float = 0.0
    ):
        """
        Args:
            vocab_size: 어휘 크기 (0~vocab_size-1)
            min_length: 최소 시퀀스 길이
            max_length: 최대 시퀀스 길이
            noise_level: 노이즈 레벨 (0.0~1.0)
        """
        self.vocab_size = vocab_size
        self.min_length = min_length
        self.max_length = max_length
        self.noise_level = noise_level
    
    def generate_sequence(self, length: int) -> List[int]:
        """랜덤 시퀀스 생성"""
        return [random.randint(1, self.vocab_size - 1) for _ in range(length)]
    
    def add_noise(self, sequence: List[int]) -> List[int]:
        """노이즈 추가 (랜덤 플립)"""
        if self.noise_level == 0.0:
            return sequence
        
        noisy_seq = sequence.copy()
        num_flips = int(len(sequence) * self.noise_level)
        flip_indices = random.sample(range(len(sequence)), num_flips)
        
        for idx in flip_indices:
            noisy_seq[idx] = random.randint(1, self.vocab_size - 1)
        
        return noisy_seq
    
    def generate(self, length: Optional[int] = None) -> TaskSample:
        """샘플 생성 (서브클래스에서 구현)"""
        raise NotImplementedError


class CopyTask(TaskGenerator):
    """Sequence Copy Task: 입력을 그대로 복사"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_id = 0
    
    def generate(self, length: Optional[int] = None) -> TaskSample:
        """Copy task 샘플 생성"""
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        input_seq = self.generate_sequence(length)
        input_seq = self.add_noise(input_seq)
        target_seq = input_seq.copy()
        
        return TaskSample(
            input=input_seq,
            target=target_seq,
            task_id=self.task_id,
            meta={
                "length": length,
                "difficulty": "easy" if length <= 10 else "hard",
                "noise_level": self.noise_level,
                "task_type": "copy"
            }
        )


class ReverseTask(TaskGenerator):
    """Reverse Task: 입력을 역순으로 변환"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_id = 1
    
    def generate(self, length: Optional[int] = None) -> TaskSample:
        """Reverse task 샘플 생성"""
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        input_seq = self.generate_sequence(length)
        input_seq = self.add_noise(input_seq)
        target_seq = input_seq[::-1]
        
        return TaskSample(
            input=input_seq,
            target=target_seq,
            task_id=self.task_id,
            meta={
                "length": length,
                "difficulty": "easy" if length <= 10 else "hard",
                "noise_level": self.noise_level,
                "task_type": "reverse"
            }
        )


class SortTask(TaskGenerator):
    """Sort Task: 입력을 오름차순으로 정렬"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_id = 2
    
    def generate(self, length: Optional[int] = None) -> TaskSample:
        """Sort task 샘플 생성"""
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        input_seq = self.generate_sequence(length)
        input_seq = self.add_noise(input_seq)
        target_seq = sorted(input_seq)
        
        return TaskSample(
            input=input_seq,
            target=target_seq,
            task_id=self.task_id,
            meta={
                "length": length,
                "difficulty": "easy" if length <= 10 else "hard",
                "noise_level": self.noise_level,
                "task_type": "sort"
            }
        )


class DelayedAssociativeRecallTask(TaskGenerator):
    """
    Delayed Associative Recall Task
    키-값 쌍을 학습하고 지연 후 키에 대한 값 회상
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_id = 3
        self.separator = 0  # 구분자 토큰
    
    def generate(self, length: Optional[int] = None) -> TaskSample:
        """Delayed Associative Recall task 샘플 생성"""
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        # 키-값 쌍 생성
        num_pairs = length // 2
        pairs = []
        for _ in range(num_pairs):
            key = random.randint(1, self.vocab_size - 1)
            value = random.randint(1, self.vocab_size - 1)
            pairs.append((key, value))
        
        # 입력: key1, value1, key2, value2, ..., query_key
        input_seq = []
        for key, value in pairs:
            input_seq.extend([key, value])
        
        # 쿼리 키 (마지막 쌍의 키)
        query_key = pairs[-1][0]
        input_seq.append(self.separator)
        input_seq.append(query_key)
        
        # 타겟: 쿼리 키에 해당하는 값
        target_seq = [pairs[-1][1]]
        
        # 노이즈 추가
        input_seq = self.add_noise(input_seq)
        
        return TaskSample(
            input=input_seq,
            target=target_seq,
            task_id=self.task_id,
            meta={
                "length": len(input_seq),
                "difficulty": "easy" if length <= 10 else "hard",
                "noise_level": self.noise_level,
                "task_type": "delayed_associative_recall",
                "num_pairs": num_pairs
            }
        )


def generate_dataset(
    task_generator: TaskGenerator,
    num_samples: int,
    split: str = "train"
) -> List[TaskSample]:
    """
    데이터셋 생성
    
    Args:
        task_generator: Task Generator
        num_samples: 샘플 개수
        split: 데이터 분할 (train, id_test, ood_test, hard)
        
    Returns:
        샘플 리스트
    """
    samples = []
    
    if split == "train":
        # Train: length 5-20
        task_generator.min_length = 5
        task_generator.max_length = 20
        task_generator.noise_level = 0.0
    elif split == "id_test":
        # ID Test: length 5-20 (train과 동일)
        task_generator.min_length = 5
        task_generator.max_length = 20
        task_generator.noise_level = 0.0
    elif split == "ood_test":
        # OOD Test: length 40-100
        task_generator.min_length = 40
        task_generator.max_length = 100
        task_generator.noise_level = 0.0
    elif split == "hard":
        # HARD: length 20-50, noise 10%
        task_generator.min_length = 20
        task_generator.max_length = 50
        task_generator.noise_level = 0.1
    
    for _ in range(num_samples):
        length = random.randint(
            task_generator.min_length,
            task_generator.max_length
        )
        sample = task_generator.generate(length=length)
        samples.append(sample)
    
    return samples

