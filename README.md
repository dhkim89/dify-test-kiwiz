# Workflow Node Tester

워크플로우의 각 노드를 개별적으로 테스트할 수 있는 API 서버입니다.

## 설치 및 실행

1. **의존성 설치:**
   ```bash
   pip install -r requirements.txt
   ```

2. **서버 실행 (간단한 테스터 - 권장):**
   ```bash
   python run_simple.py
   ```
   - 포트: `http://localhost:8003`
   - Swagger UI: `http://localhost:8003/docs`

3. **서버 실행 (전체 워크플로우 - 실험적):**
   ```bash
   python run.py
   ```
   - 포트: `http://localhost:8002`
   - 주의: 의존성 문제로 일부 노드가 동작하지 않을 수 있습니다.

## API 엔드포인트

### 기본 정보
- `GET /` - API 상태 및 엔드포인트 정보
- `GET /node-types` - 사용 가능한 노드 타입 목록
- `GET /node-examples/{node_type}` - 특정 노드의 예제 설정

### 노드 테스트
- `POST /test-node` - 단일 노드 테스트
- `POST /run-node` - 완전한 워크플로우 실행

## 포스트맨 테스트

### 간단한 테스터용 (권장)
1. **포스트맨 컬렉션 import:**
   - `simple_postman_collection.json` 파일을 포스트맨에서 import

2. **환경 변수 설정:**
   - `base_url`: `http://localhost:8003`

### 전체 워크플로우용 (실험적)
1. **포스트맨 컬렉션 import:**
   - `postman_collection.json` 파일을 포스트맨에서 import

2. **환경 변수 설정:**
   - `base_url`: `http://localhost:8002`

## 지원하는 노드 타입

### 간단한 테스터에서 지원하는 노드 (동작 확인됨)
- **code**: Python 코드 실행 - ✅ 테스트 완료
- **template_transform**: 템플릿 변환 - ✅ 테스트 완료  
- **http_request**: HTTP 요청 - ✅ 테스트 완료
- **variable_assigner**: 변수 할당 - ✅ 테스트 완료
- **if_else**: 조건 분기 - ✅ 테스트 완료

### 전체 워크플로우에서 지원 예정 (실험적)
- **llm**: LLM 모델 호출
- **parameter_extractor**: 파라미터 추출
- **knowledge_retrieval**: 지식 검색
- **tool**: 도구 사용
- **start**: 시작 노드
- **end**: 종료 노드
- **answer**: 답변 노드

## 테스트 예제

### 1. 코드 노드 테스트 (✅ 동작 확인됨)

```bash
curl -X POST "http://localhost:8003/test-node" \
  -H "Content-Type: application/json" \
  -d '{
    "node_type": "code",
    "config": {
      "code": "result = inputs[\"number\"] * 2\noutput = {\"doubled\": result, \"original\": inputs[\"number\"]}",
      "code_language": "python3"
    },
    "inputs": {
      "number": 10
    }
  }'
```

**응답 예시:**
```json
{
  "status": "success",
  "node_type": "code",
  "result": {
    "doubled": 20,
    "original": 10
  },
  "error": null
}
```

### 2. 템플릿 변환 노드 테스트 (✅ 동작 확인됨)

```bash
curl -X POST "http://localhost:8003/test-node" \
  -H "Content-Type: application/json" \
  -d '{
    "node_type": "template_transform",
    "config": {
      "template": "안녕하세요, {{name}}님! 오늘은 {{date}}입니다."
    },
    "inputs": {
      "name": "홍길동",
      "date": "2024-01-15"
    }
  }'
```

**응답 예시:**
```json
{
  "status": "success",
  "node_type": "template_transform", 
  "result": {
    "output": "안녕하세요, 홍길동님! 오늘은 2024-01-15입니다."
  },
  "error": null
}
```

## 주요 성과

✅ **성공적으로 구축된 기능들:**
- 간단한 워크플로우 노드 테스터 API 서버
- 5가지 핵심 노드 타입 지원 (code, template_transform, http_request, variable_assigner, if_else)
- 포스트맨 컬렉션을 통한 즉시 테스트 가능
- 실제 동작 확인 완료

## 주의사항

- **간단한 테스터 (`simple_main.py`)**: 의존성 문제 없이 즉시 사용 가능 ✅
- **전체 워크플로우 (`main.py`)**: 복잡한 의존성으로 인해 추가 설정 필요
- HTTP 요청 노드는 외부 네트워크 접근 필요
- 코드 실행 노드는 보안상 제한된 Python 환경에서 실행됨

## 트러블슈팅

### 의존성 오류
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 포트 변경
간단한 테스터의 포트 변경:
```python
# run_simple.py 파일에서
uvicorn.run("simple_main:app", host="0.0.0.0", port=8004, reload=True)
```

### 서버 접속 확인
```bash
curl http://localhost:8003/
curl http://localhost:8003/node-types
```