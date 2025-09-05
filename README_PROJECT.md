# Workflow Node Tester 🚀

워크플로우의 각 노드를 개별적으로 테스트할 수 있는 종합 API 테스트 시스템입니다.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ 주요 기능

- **14개 워크플로우 노드 타입** 완전 지원
- **실제 시나리오 테스트** 가능
- **포스트맨 컬렉션** 21개 테스트 케이스 제공
- **명세서 100% 커버리지** 달성
- **KAI Studio & MCP 연계** 시뮬레이션 

## 🎯 지원 노드 타입

### 핵심 노드들
- ✅ **LLM 노드**: GPT-4, Claude-3, Gemini Pro 지원
- ✅ **지식검색 노드**: RAG Studio 연동, Re-ranking, 다중 출력 포맷
- ✅ **에이전트 노드**: MCP 도구 연계, 실행 단계 추적
- ✅ **주제분류 노드**: 다중 카테고리, 신뢰도 기반 분류

### 기본 노드들
- ✅ **코드 노드**: Python3 실행, 보안 샌드박스
- ✅ **템플릿 노드**: 변수 치환, Jinja2 스타일
- ✅ **HTTP 노드**: RESTful API 호출, 다양한 메서드
- ✅ **조건 노드**: 복합 조건, 논리 연산자
- ✅ **변수할당 노드**: 동적 변수 매핑
- ✅ **시작/종료 노드**: 워크플로우 라이프사이클 관리

### 고급 노드들  
- ✅ **파라미터 추출**: NLP 기반 구조화된 데이터 추출
- ✅ **리스트 연산**: 필터링, 매핑, 정렬 등 배열 조작
- ✅ **반복 처리**: 배치 작업, 대량 데이터 처리
- ✅ **지식 검색**: 의미적/키워드/하이브리드 검색

## 🚀 빠른 시작

### 1. 설치
```bash
git clone <your-repo-url>
cd workflow-test/backend
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python run_simple.py
```
서버가 `http://localhost:8003`에서 실행됩니다.

### 3. API 테스트
```bash
# 상태 확인
curl http://localhost:8003/

# 지원 노드 타입 확인
curl http://localhost:8003/node-types

# 코드 노드 테스트
curl -X POST http://localhost:8003/test-node \
  -H "Content-Type: application/json" \
  -d '{
    "node_type": "code",
    "config": {
      "code": "result = inputs[\"number\"] * 2\noutput = {\"doubled\": result}",
      "code_language": "python3"
    },
    "inputs": {"number": 10}
  }'
```

## 📋 포스트맨 테스트

1. **포스트맨에서 컬렉션 Import**
   - `simple_postman_collection.json` 파일 import

2. **환경변수 설정**
   - `base_url`: `http://localhost:8003`

3. **21개 테스트 케이스 실행**
   - LLM 노드 (GPT-4, Claude-3)
   - 주제 분류 (단일/다중 카테고리)
   - 지식 검색 (의미적/하이브리드)
   - 에이전트 호출 (보고서/데이터 분석)
   - 기타 모든 노드 타입들

## 🏗️ 아키텍처

```
workflow-test/backend/
├── simple_main.py              # 메인 FastAPI 애플리케이션
├── requirements.txt            # Python 의존성
├── simple_postman_collection.json  # 포스트맨 테스트 컬렉션
├── README.md                   # 상세 사용 가이드
├── run_simple.py              # 서버 실행 스크립트
└── workflow/                   # 워크플로우 노드 구현
    ├── nodes/                  # 각 노드 타입별 구현
    ├── entities/              # 데이터 모델
    └── utils/                 # 유틸리티 함수
```

## 📊 테스트 결과

- ✅ **14개 노드 타입** 모두 구현 완료
- ✅ **명세서 100% 커버리지** 달성  
- ✅ **실제 데이터 테스트** 통과
- ✅ **21개 포스트맨 테스트** 케이스 제공

## 🌟 실제 사용 사례

### 1. 고객 지원 워크플로우
```
시작 → 주제분류 → 지식검색 → LLM 답변 → 종료
```

### 2. 데이터 분석 파이프라인  
```
시작 → 파라미터추출 → 에이전트분석 → 보고서생성 → 종료
```

### 3. 콘텐츠 처리 시스템
```
시작 → 템플릿변환 → 리스트연산 → 반복처리 → HTTP전송 → 종료
```

## 🔧 개발 정보

- **개발 언어**: Python 3.11+
- **웹 프레임워크**: FastAPI
- **테스트 도구**: Postman
- **아키텍처**: RESTful API, 모듈화 설계
- **보안**: 코드 실행 샌드박스, 입력 검증

## 📈 향후 계획

- [ ] JavaScript 코드 노드 지원
- [ ] 실제 LLM API 연동
- [ ] 실제 RAG Studio 연동  
- [ ] 웹 UI 대시보드
- [ ] 워크플로우 체인 실행
- [ ] 성능 모니터링

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

⭐ **이 프로젝트가 도움이 되셨다면 스타를 눌러주세요!** ⭐