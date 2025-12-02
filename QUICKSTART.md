# BioHama PoC 빠른 시작 가이드

## 설치

```bash
pip install -r requirements.txt
```

## 간단한 실험 실행

Hydra 없이 빠르게 테스트하려면:

```bash
python experiments/run_experiment.py
```

## 전체 실험 실행 (Hydra 사용)

```bash
cd experiments
python main.py
```

다른 모델로 실험:

```bash
python main.py model=lstm
python main.py model=transformer
python main.py model=gru
```

다른 태스크로 실험:

```bash
python main.py task=reverse
python main.py task=sort
```

## 테스트 실행

```bash
python -m pytest tests/
```

또는

```bash
python tests/test_biohama.py
```

## 프로젝트 구조

```
biohama-project/
├── biohama/              # Core 모듈
│   ├── modules/         # Router, WM, Attention
│   ├── mechanism/       # Hebbian, Message Passing
│   └── biohama_model.py
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

## 주요 기능

- **Router**: 동적 모듈 선택
- **Working Memory**: 장거리 의존성 처리
- **Hebbian Learning**: 모듈 간 연결 학습
- **Message Passing**: 모듈 간 정보 전달
- **Sparse Attention**: 효율적인 주의 메커니즘

## 성공 조건

- Train Seq-Acc > 95%
- ID Test Seq-Acc > 90%
- OOD Test Seq-Acc > 70% (핵심!)
- HARD Test 성능 우수

