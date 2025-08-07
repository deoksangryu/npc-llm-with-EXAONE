# 🎮 NPC 대화 시스템 v1 Final - LLM 기반 혁신적 AI 대화 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-green.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 프로젝트 소개

**NPC 대화 시스템 v1 Final**은 LLM(Large Language Model) 기반의 혁신적인 NPC 대화 시스템입니다. EXAONE 3.5 7.8B 모델을 활용하여 자연스러운 대화와 맞춤형 퀘스트 생성이 가능한 게임 AI 시스템입니다.

### 🎯 핵심 혁신

- **🧠 AdvancedAnalyzer**: LLM 기반 3단계 의도 분석 시스템
- **💬 자연스러운 대화**: 친밀도 기반 동적 말투 변화
- **📋 맞춤형 퀘스트**: 사용자 요구사항 반영한 다양한 퀘스트 생성
- **🔍 RAG 시스템**: 게임 세계관 정보 검색 및 활용
- **🎭 NPC 개성**: 종족, 성별, 나이별 고유한 말투 구현

### 🏆 달성 성과

- **Identity Question 정확도**: 20% → 80% (4배 개선!)
- **퀘스트 다양성**: 0% → 100% (동일 퀘스트 반복 완전 해결!)
- **사용자 만족도**: 의도 정확히 파악된 맞춤형 응답
- **시스템 안정성**: LLM 장애 시에도 기본 기능 보장

## 🚀 주요 기능

### 1. 혁신적 의도 분석 시스템
```python
# 3단계 분석 프로세스
STEP 1: Identity Question 판별
STEP 2: 요청 유형 판별  
STEP 3: 세부사항 추출
```

### 2. 친밀도 기반 동적 대화
- **6단계 친밀도 시스템**: -1.0 ~ 1.0 범위
- **감정 기반 업데이트**: positive/negative/neutral에 따른 동적 변화
- **맞춤형 말투**: 친밀도에 따른 자연스러운 대화 톤 변화

### 3. RAG 기반 게임 정보 검색
- **4개 데이터 소스**: lore, npc, monster, quest
- **FAISS 벡터 검색**: 고속 유사도 검색
- **맥락 제공**: 사용자 질문과 관련된 게임 정보 자동 추출

### 4. 맞춤형 퀘스트 생성
- **사용자 요구사항 반영**: 난이도, 테마, 긴급도별 맞춤 생성
- **다양성 보장**: 동일 퀘스트 반복 방지
- **시각화**: CLI에서 게임 UI 같은 퀘스트 표시

## 🛠️ 기술 스택

### Core Technologies
- **Python 3.8+**: 메인 프로그래밍 언어
- **PyTorch 2.0+**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 트랜스포머 라이브러리
- **EXAONE 3.5 7.8B**: 한국어 성능 우수한 LLM 모델

### AI/ML Libraries
- **Sentence Transformers**: 텍스트 임베딩
- **FAISS**: 벡터 검색 엔진
- **BitsAndBytes**: 4bit 양자화

### Data Processing
- **JSON**: 데이터 구조화
- **Regular Expressions**: 텍스트 처리
- **Dataclasses**: 데이터 클래스 정의

## 📦 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/deoksangryu/npc-llm-with-EXAONE.git
cd npc-llm-with-EXAONE
```

### 2. 의존성 설치
```bash
pip install torch transformers sentence-transformers faiss-cpu bitsandbytes
```

### 3. 데이터 디렉토리 생성
```bash
mkdir data
# 필요한 경우 게임 데이터 파일들을 data/ 디렉토리에 추가
```

## 🎮 사용 방법

### 1. 기본 실행
```bash
python npc_llm_v1_final.py
```

### 2. NPC 선택
시스템이 시작되면 대화할 NPC를 선택할 수 있습니다:
- **리나 인버스**: 장난기 많고 호기심 많은 마법사
- **제르가디스**: 냉정하고 신중한 키메라 검사  
- **아멜리아**: 정의감이 강하고 밝은 공주

### 3. 사용자 프로필 설정
- **이름**: 사용자 정의 이름 설정
- **레벨**: 게임 진행도 반영

### 4. 대화 명령어
- `quit`, `exit`, `종료`: 시스템 종료
- `debug on/off`: 추론 과정 표시 토글
- `status`: 현재 친밀도 확인

## 🏗️ 시스템 아키텍처

### 핵심 컴포넌트

#### 1. AdvancedAnalyzer
```python
class AdvancedAnalyzer:
    """
    LLM 기반 3단계 의도 분석 시스템
    - Identity Question 판별
    - 요청 유형 분류
    - 세부사항 추출
    """
```

#### 2. RAGMemory
```python
class RAGMemory:
    """
    게임 세계관 정보 검색 엔진
    - FAISS 벡터 검색
    - 4개 데이터 소스 통합
    - 맥락 기반 정보 제공
    """
```

#### 3. NPCDialogueAgentV1Final
```python
class NPCDialogueAgentV1Final:
    """
    메인 대화 엔진
    - 의도 분석과 응답 생성 통합
    - 친밀도 시스템 관리
    - 세션 히스토리 유지
    """
```

### 데이터 흐름
```
사용자 입력 → 의도 분석 → 친밀도 업데이트 → 프롬프트 생성 → LLM 응답 → 결과 통합
```

## 📊 성능 지표

### 정확도 개선
- **Identity Question**: 20% → 80% (4배 개선)
- **퀘스트 다양성**: 0% → 100% (완전 해결)
- **의도 파악**: 키워드 매칭 → 의미적 이해

### 시스템 안정성
- **폴백 시스템**: LLM 실패 시 키워드 기반 분석
- **오류 복구**: 각 단계별 예외 처리
- **메모리 효율성**: 4bit 양자화 적용

## 🔧 개발 환경

### 지원 플랫폼
- **macOS**: MPS 가속 지원
- **Windows**: CUDA 가속 지원
- **Linux**: CPU/GPU 가속 지원

### 하드웨어 요구사항
- **최소**: 8GB RAM
- **권장**: 16GB+ RAM, GPU 지원
- **저장공간**: 10GB+ (모델 포함)

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👨‍💻 개발자

**류덕상** - [GitHub](https://github.com/deoksangryu)

## 🙏 감사의 말

- **LG AI Research**: EXAONE 모델 제공
- **Hugging Face**: Transformers 라이브러리
- **Facebook Research**: FAISS 벡터 검색 엔진

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 [Issues](https://github.com/deoksangryu/npc-llm-with-EXAONE/issues)를 통해 연락해주세요.

---

⭐ 이 프로젝트가 도움이 되었다면 **Star**를 눌러주세요! 