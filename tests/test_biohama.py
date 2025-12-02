"""
BioHama 모델 기본 테스트
"""

import torch
from biohama.biohama_model import BioHamaModel
from biohama.common import TaskSample


def test_biohama_forward():
    """BioHama 모델 forward 테스트"""
    model = BioHamaModel(
        vocab_size=100,
        embedding_dim=64,
        num_modules=4,
        memory_size=8
    )
    
    batch_size = 2
    seq_len = 10
    input_seq = torch.randint(1, 100, (batch_size, seq_len))
    task_id = 0
    
    logits = model.forward(input_seq, task_id)
    
    assert logits.shape == (batch_size, seq_len, 100), \
        f"Expected shape (2, 10, 100), got {logits.shape}"
    
    print("✓ BioHama forward test passed")


def test_biohama_train_step():
    """BioHama 모델 train_step 테스트"""
    model = BioHamaModel(
        vocab_size=100,
        embedding_dim=64,
        num_modules=4,
        memory_size=8
    )
    
    batch = [
        TaskSample(
            input=[1, 2, 3, 4, 5],
            target=[1, 2, 3, 4, 5],
            task_id=0,
            meta={}
        ),
        TaskSample(
            input=[5, 4, 3, 2, 1],
            target=[5, 4, 3, 2, 1],
            task_id=0,
            meta={}
        )
    ]
    
    loss, metrics = model.train_step(batch, device='cpu')
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert 'loss' in metrics, "Metrics should contain 'loss'"
    assert 'token_acc' in metrics, "Metrics should contain 'token_acc'"
    assert 'seq_acc' in metrics, "Metrics should contain 'seq_acc'"
    
    print("✓ BioHama train_step test passed")


if __name__ == "__main__":
    test_biohama_forward()
    test_biohama_train_step()
    print("\nAll tests passed!")

