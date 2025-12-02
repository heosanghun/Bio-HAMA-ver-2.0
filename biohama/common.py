"""
공통 인터페이스 및 데이터 포맷 정의
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class TaskSample:
    """표준 Task 샘플 포맷"""
    input: List[int]  # Token IDs
    target: List[int]  # Target sequence
    task_id: int  # 0:Copy, 1:Reverse, 2:Sort, 3:DelayedAssociativeRecall
    meta: Dict[str, Any]  # 추가 메타데이터
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화용 딕셔너리 변환"""
        return {
            "input": self.input,
            "target": self.target,
            "task_id": self.task_id,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSample':
        """딕셔너리에서 TaskSample 생성"""
        return cls(
            input=data["input"],
            target=data["target"],
            task_id=data["task_id"],
            meta=data.get("meta", {})
        )
    
    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TaskSample':
        """JSON 문자열에서 TaskSample 생성"""
        return cls.from_dict(json.loads(json_str))


class BaseModel:
    """모든 모델의 공통 인터페이스"""
    
    def forward(self, input_seq: List[int], task_id: int) -> List[int]:
        """
        모델의 forward pass
        
        Args:
            input_seq: 입력 시퀀스 (Token IDs)
            task_id: Task ID
            
        Returns:
            예측된 출력 시퀀스
        """
        raise NotImplementedError
    
    def train_step(self, batch: List[TaskSample], device: str = 'cpu') -> tuple:
        """학습 스텝 - (loss_tensor, metrics_dict) 반환"""
        raise NotImplementedError
    
    def eval_step(self, batch: List[TaskSample]) -> Dict[str, float]:
        """평가 스텝"""
        raise NotImplementedError

