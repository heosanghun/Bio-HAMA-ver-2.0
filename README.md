# BioHama: Bio-inspired Hierarchical Attention Memory Architecture

뇌과학에서 영감을 받은 계층적 주의 메모리 아키텍처로, OOD(Out-of-Distribution) 상황에서의 일반화 능력을 목표로 합니다.

## 📋 개요

BioHama는 NeurIPS 등재를 목표로 한 연구 프로젝트로, 다음과 같은 핵심 구성요소를 포함합니다:

- **Router**: 동적 모듈 선택 메커니즘
- **Working Memory**: 장거리 의존성을 처리하는 동적 메모리 시스템
- **Hebbian Learning**: "Neurons that fire together, wire together" 원칙 기반 학습
- **Message Passing**: 모듈 간 정보 전달 메커니즘
- **Sparse Attention**: 효율적인 주의 메커니즘

## 🚀 빠른 시작

### 설치

```bash
pip install -r requirements.txt
```

### 간단한 실험 실행

```bash
python experiments/run_experiment.py
```

### 전체 실험 실행 (Hydra)

```bash
cd experiments
python main.py
```

더 자세한 내용은 [QUICKSTART.md](QUICKSTART.md)를 참조하세요.

## 📁 프로젝트 구조

```
biohama-project/
├── biohama/              # Core 모듈
│   ├── modules/         # Router, WM, Attention
│   ├── mechanism/       # Hebbian, Message Passing
│   ├── common.py        # 공통 인터페이스
│   └── biohama_model.py # 통합 모델
├── experiments/         # 실험 실행
│   ├── tasks/          # Task Generators
│   ├── baselines/      # LSTM, Transformer, GRU
│   ├── trainer.py      # 학습/평가 Trainer
│   ├── main.py         # Hydra 메인 스크립트
│   └── run_experiment.py  # 간단한 실행 스크립트
├── configs/            # Hydra 설정 파일
│   ├── model/
│   └── task/
└── tests/              # Unit Tests
```

## 🎯 실험 태스크

- **Copy**: 시퀀스 복사
- **Reverse**: 시퀀스 역순 변환
- **Sort**: 시퀀스 정렬
- **Delayed Associative Recall**: 지연 연상 회상

## 📊 평가 지표

- **Sequence Accuracy (Seq-Acc)**: 전체 시퀀스 정확도 (최우선 지표)
- **Token Accuracy**: 개별 토큰 정확도
- **Perplexity**: 예측 불확실성

## ✅ 성공 조건

- Train Seq-Acc > 95%
- ID Test Seq-Acc > 90%
- OOD Test Seq-Acc > 70% (핵심 차별점!)
- HARD Test에서 Baseline 대비 20%p 이상 우수

## 🔬 Baseline 모델

- LSTM
- GRU
- Transformer (Decoder-only)

## 📚 참고 문헌

- [2106.08170](https://arxiv.org/pdf/2106.08170)
- [2310.18777](https://arxiv.org/pdf/2310.18777)
- [2412.14076](https://arxiv.org/pdf/2412.14076)
- NeurIPS 2023 Clear Continual Learning

## ✅ 테스트 결과

### 기본 테스트 통과

모든 핵심 구성요소가 정상 동작함을 확인했습니다:

1. **Router 테스트** ✓
   - 모듈 선택 메커니즘 정상 동작
   - 출력 shape: [batch_size, num_modules]

2. **Working Memory 테스트** ✓
   - 메모리 쓰기/읽기 정상 동작
   - Attention 기반 메모리 관리 정상

3. **BioHama 모델 테스트** ✓
   - Forward pass 정상 동작
   - 출력 shape: [batch_size, seq_len, vocab_size]
   - 모델 파라미터 수: 약 714,420개

4. **Train step 테스트** ✓
   - Loss 계산 정상
   - Metrics 계산 정상

5. **Task Generator 테스트** ✓
   - Copy Task 생성 정상
   - 데이터셋 생성 정상

6. **데이터셋 생성 테스트** ✓
   - Train/ID/OOD/HARD split 정상 생성

### 테스트 실행 방법

```bash
# 빠른 테스트
python test_quick.py

# 또는 pytest 사용
python -m pytest tests/
```

### 테스트 결과 예시

```
============================================================
BioHama PoC 빠른 테스트
============================================================

1. Router 테스트...
   ✓ Router 출력 shape: torch.Size([2, 8])

2. Working Memory 테스트...
   ✓ Working Memory write/read 성공

3. BioHama 모델 테스트...
   ✓ 모델 forward 성공
   ✓ 출력 shape: torch.Size([2, 10, 100])

4. Train step 테스트...
   ✓ Train step 성공
   ✓ Loss: 4.6251
   ✓ Token Acc: 0.0000
   ✓ Seq Acc: 0.0000

5. Task Generator 테스트...
   ✓ Task 생성 성공

6. 데이터셋 생성 테스트...
   ✓ 데이터셋 생성 성공: 10 samples

============================================================
모든 테스트 통과! ✓
============================================================
```

## 📝 라이선스

연구 목적으로 사용 가능합니다.

