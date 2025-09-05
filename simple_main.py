from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import logging
import requests
import json

app = FastAPI(title="Simple Workflow Node Tester", description="간단한 워크플로우 노드 테스트 API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeTestRequest(BaseModel):
    node_type: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]

class NodeTestResponse(BaseModel):
    status: str
    node_type: str
    result: Dict[str, Any]
    error: str = None

def execute_code_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python 코드 실행 노드"""
    try:
        code = config.get('code', '')
        code_language = config.get('code_language', 'python3')
        
        if code_language != 'python3':
            raise ValueError(f"Unsupported language: {code_language}")
        
        # 안전한 실행을 위한 글로벌 네임스페이스
        globals_dict = {
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'dict': dict,
                'list': list,
                'print': print,
            }
        }
        
        locals_dict = {'inputs': inputs}
        
        exec(code, globals_dict, locals_dict)
        
        return locals_dict.get('output', {})
        
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        raise Exception(f"Code execution failed: {str(e)}")

def execute_template_transform_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """템플릿 변환 노드"""
    try:
        template = config.get('template', '')
        
        # 간단한 템플릿 변환 ({{variable}} 형태)
        result = template
        for key, value in inputs.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        
        return {'output': result}
        
    except Exception as e:
        logger.error(f"Template transform error: {e}")
        raise Exception(f"Template transform failed: {str(e)}")

def execute_http_request_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """HTTP 요청 노드"""
    try:
        method = config.get('method', 'GET').upper()
        url = config.get('url')
        headers = config.get('headers', {})
        
        if not url:
            raise ValueError("URL is required")
        
        # URL에서 변수 치환
        for key, value in inputs.items():
            url = url.replace(f"{{{{{key}}}}}", str(value))
        
        response = requests.request(method, url, headers=headers, timeout=10)
        
        try:
            json_response = response.json()
        except:
            json_response = response.text
            
        return {
            'status_code': response.status_code,
            'body': json_response,
            'headers': dict(response.headers)
        }
        
    except Exception as e:
        logger.error(f"HTTP request error: {e}")
        raise Exception(f"HTTP request failed: {str(e)}")

def execute_variable_assigner_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """변수 할당 노드"""
    try:
        assignments = config.get('assignments', [])
        output = {}
        
        for assignment in assignments:
            var_name = assignment.get('variable')
            var_value = assignment.get('value')
            
            if not var_name:
                continue
            
            # 변수 치환
            if isinstance(var_value, str):
                for key, value in inputs.items():
                    var_value = var_value.replace(f"{{{{{key}}}}}", str(value))
            
            output[var_name] = var_value
        
        return output
        
    except Exception as e:
        logger.error(f"Variable assignment error: {e}")
        raise Exception(f"Variable assignment failed: {str(e)}")

def execute_if_else_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """조건 분기 노드"""
    try:
        conditions = config.get('conditions', [])
        logical_operator = config.get('logical_operator', 'and')
        
        results = []
        
        for condition in conditions:
            variable = condition.get('variable')
            operator = condition.get('comparison_operator')
            expected_value = condition.get('value')
            
            actual_value = inputs.get(variable)
            
            if operator == 'eq':
                result = actual_value == expected_value
            elif operator == 'ne':
                result = actual_value != expected_value
            elif operator == 'gt':
                result = actual_value > expected_value
            elif operator == 'gte':
                result = actual_value >= expected_value
            elif operator == 'lt':
                result = actual_value < expected_value
            elif operator == 'lte':
                result = actual_value <= expected_value
            else:
                result = False
            
            results.append(result)
        
        if logical_operator == 'and':
            final_result = all(results)
        else:  # or
            final_result = any(results)
        
        return {
            'result': final_result,
            'branch': 'true' if final_result else 'false'
        }
        
    except Exception as e:
        logger.error(f"If-else condition error: {e}")
        raise Exception(f"If-else condition failed: {str(e)}")

def execute_parameter_extractor_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """파라미터 추출 노드"""
    try:
        parameters = config.get('parameters', [])
        text_input = inputs.get('text', '')
        
        output = {}
        
        for param in parameters:
            param_name = param.get('name')
            param_type = param.get('type', 'string')
            required = param.get('required', False)
            description = param.get('description', '')
            
            # 간단한 파라미터 추출 로직 (실제로는 더 복잡한 NLP 처리 필요)
            if param_name.lower() in text_input.lower():
                if param_type == 'number':
                    # 숫자 추출 시도
                    import re
                    numbers = re.findall(r'\d+', text_input)
                    output[param_name] = int(numbers[0]) if numbers else None
                elif param_type == 'boolean':
                    output[param_name] = 'yes' in text_input.lower() or 'true' in text_input.lower()
                else:
                    # 단순 텍스트 추출
                    words = text_input.split()
                    for i, word in enumerate(words):
                        if param_name.lower() in word.lower() and i + 1 < len(words):
                            output[param_name] = words[i + 1]
                            break
                    else:
                        output[param_name] = text_input[:50]  # 첫 50자
            else:
                if required:
                    output[param_name] = None
                else:
                    output[param_name] = f"Parameter '{param_name}' not found in text"
        
        return output
        
    except Exception as e:
        logger.error(f"Parameter extraction error: {e}")
        raise Exception(f"Parameter extraction failed: {str(e)}")

def execute_start_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """시작 노드"""
    try:
        return {
            'status': 'started',
            'inputs_received': inputs,
            'timestamp': json.dumps(inputs)  # JSON serialization test
        }
        
    except Exception as e:
        logger.error(f"Start node error: {e}")
        raise Exception(f"Start node failed: {str(e)}")

def execute_end_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """종료 노드"""
    try:
        output_config = config.get('outputs', {})
        
        result = {
            'status': 'completed',
            'final_outputs': {}
        }
        
        # 설정된 출력 매핑에 따라 결과 생성
        for output_key, input_key in output_config.items():
            if input_key in inputs:
                result['final_outputs'][output_key] = inputs[input_key]
            else:
                result['final_outputs'][output_key] = f"Input '{input_key}' not found"
        
        # 설정이 없으면 모든 입력을 그대로 출력
        if not output_config:
            result['final_outputs'] = inputs
            
        return result
        
    except Exception as e:
        logger.error(f"End node error: {e}")
        raise Exception(f"End node failed: {str(e)}")

def execute_list_operator_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """리스트 연산자 노드"""
    try:
        operation = config.get('operation', 'append')
        list_input = inputs.get('list', [])
        item = inputs.get('item')
        
        if not isinstance(list_input, list):
            list_input = [list_input]
            
        result_list = list_input.copy()
        
        if operation == 'append':
            result_list.append(item)
        elif operation == 'prepend':
            result_list.insert(0, item)
        elif operation == 'remove' and item in result_list:
            result_list.remove(item)
        elif operation == 'filter':
            # 간단한 필터링 (특정 값과 일치하는 항목만 유지)
            filter_value = config.get('filter_value')
            result_list = [x for x in result_list if x == filter_value]
        elif operation == 'map':
            # 간단한 매핑 (모든 항목에 접두사 추가)
            prefix = config.get('map_prefix', 'mapped_')
            result_list = [f"{prefix}{x}" for x in result_list]
        elif operation == 'sort':
            try:
                result_list.sort()
            except:
                pass  # 정렬할 수 없는 경우 그대로 유지
                
        return {
            'result_list': result_list,
            'operation_applied': operation,
            'original_length': len(list_input),
            'result_length': len(result_list)
        }
        
    except Exception as e:
        logger.error(f"List operator error: {e}")
        raise Exception(f"List operator failed: {str(e)}")

def execute_iteration_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """반복 노드"""
    try:
        items = inputs.get('items', [])
        max_iterations = config.get('max_iterations', 10)
        
        if not isinstance(items, list):
            items = [items]
            
        results = []
        
        for i, item in enumerate(items[:max_iterations]):
            iteration_result = {
                'iteration_index': i,
                'current_item': item,
                'processed': f"Processed: {item}"
            }
            results.append(iteration_result)
            
        return {
            'total_iterations': len(results),
            'results': results,
            'completed': True
        }
        
    except Exception as e:
        logger.error(f"Iteration error: {e}")
        raise Exception(f"Iteration failed: {str(e)}")

def execute_knowledge_retrieval_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """지식 검색 노드 (모의 RAG Studio 연동)"""
    try:
        query = inputs.get('query', '')
        dataset_id = config.get('dataset_id', 'default')
        top_k = config.get('top_k', 3)
        output_format = config.get('output_format', 'text')  # text, json, structured
        search_mode = config.get('search_mode', 'semantic')  # semantic, keyword, hybrid
        
        # 모의 지식 데이터베이스 (RAG Studio 데이터 시뮬레이션)
        mock_knowledge = [
            {"id": 1, "content": "Python은 고급 프로그래밍 언어입니다. 간결한 문법과 강력한 라이브러리를 제공합니다.", "source": "python_docs.md", "score": 0.95},
            {"id": 2, "content": "FastAPI는 Python 웹 프레임워크입니다. 높은 성능과 자동 API 문서화를 지원합니다.", "source": "api_guide.pdf", "score": 0.87},
            {"id": 3, "content": "워크플로우는 작업의 흐름을 나타냅니다. 노드와 엣지로 구성되어 있습니다.", "source": "workflow_manual.docx", "score": 0.82},
            {"id": 4, "content": "API는 소프트웨어 간 상호작용을 가능하게 합니다. RESTful API가 가장 일반적입니다.", "source": "architecture_guide.md", "score": 0.78},
            {"id": 5, "content": "JSON은 데이터 교환 형식입니다. 키-값 쌍으로 구조화된 데이터를 표현합니다.", "source": "data_formats.txt", "score": 0.71}
        ]
        
        # 검색 모드에 따른 처리
        query_lower = query.lower()
        scored_results = []
        
        for item in mock_knowledge:
            content_lower = item['content'].lower()
            
            if search_mode == 'semantic':
                # 의미 기반 검색 시뮬레이션
                if any(word in content_lower for word in query_lower.split()):
                    # 의미 유사도 시뮬레이션 (실제로는 임베딩 모델 사용)
                    semantic_score = item['score'] * 0.9
                    item_copy = item.copy()
                    item_copy['relevance_score'] = semantic_score
                    scored_results.append(item_copy)
            elif search_mode == 'keyword':
                # 키워드 기반 검색
                if any(word in content_lower for word in query_lower.split()):
                    item_copy = item.copy()
                    item_copy['relevance_score'] = item['score'] * 0.8
                    scored_results.append(item_copy)
            elif search_mode == 'hybrid':
                # 하이브리드 검색 (의미 + 키워드)
                if any(word in content_lower for word in query_lower.split()):
                    item_copy = item.copy()
                    item_copy['relevance_score'] = item['score']  # 최고 점수
                    scored_results.append(item_copy)
        
        # Re-ranking 시뮬레이션
        scored_results = sorted(scored_results, key=lambda x: x['relevance_score'], reverse=True)[:top_k]
        
        # 출력 포맷에 따른 결과 가공
        if output_format == 'structured':
            formatted_results = [
                {
                    'title': f"Document {item['id']}",
                    'content': item['content'],
                    'metadata': {
                        'source': item['source'],
                        'relevance': item['relevance_score']
                    }
                }
                for item in scored_results
            ]
        elif output_format == 'json':
            formatted_results = scored_results
        else:  # text
            formatted_results = [item['content'] for item in scored_results]
        
        return {
            'query': query,
            'dataset_id': dataset_id,
            'search_mode': search_mode,
            'output_format': output_format,
            'retrieved_count': len(scored_results),
            'results': formatted_results,
            'rag_metadata': {
                'embedding_model': 'text-embedding-ada-002',
                'rerank_applied': True,
                'total_documents_searched': len(mock_knowledge)
            }
        }
        
    except Exception as e:
        logger.error(f"Knowledge retrieval error: {e}")
        raise Exception(f"Knowledge retrieval failed: {str(e)}")

def execute_llm_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 노드 (모의 KAI Studio 연동)"""
    try:
        model_name = config.get('model', 'gpt-3.5-turbo')
        prompt_template = config.get('prompt_template', '{{query}}')
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 1000)
        system_prompt = config.get('system_prompt', '')
        
        # 프롬프트 변수 치환
        prompt = prompt_template
        for key, value in inputs.items():
            placeholder = f"{{{{{key}}}}}"
            prompt = prompt.replace(placeholder, str(value))
        
        # 모의 LLM 응답 생성 (실제로는 KAI Studio API 호출)
        mock_responses = {
            'gpt-3.5-turbo': f"GPT-3.5 응답: {prompt}에 대한 답변입니다. (온도: {temperature})",
            'gpt-4': f"GPT-4 응답: {prompt}에 대한 더 정교한 답변입니다.",
            'claude-3': f"Claude-3 응답: {prompt}에 대해 신중하게 답변드립니다.",
            'gemini-pro': f"Gemini Pro 응답: {prompt}에 대한 구글의 답변입니다."
        }
        
        response_text = mock_responses.get(model_name, f"Unknown model {model_name}의 응답입니다.")
        
        return {
            'response': response_text,
            'model_used': model_name,
            'prompt_sent': prompt,
            'system_prompt': system_prompt,
            'generation_info': {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'token_count': len(response_text.split()),
                'model_version': '2024.01'
            },
            'mcp_functions_available': ['web_search', 'code_execution', 'file_operations']
        }
        
    except Exception as e:
        logger.error(f"LLM node error: {e}")
        raise Exception(f"LLM node failed: {str(e)}")

def execute_agent_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """에이전트 호출 노드 (모의 MCP 연계)"""
    try:
        agent_id = config.get('agent_id', 'default_agent')
        agent_name = config.get('agent_name', 'Custom Agent')
        mcp_tools = config.get('mcp_tools', [])
        max_iterations = config.get('max_iterations', 5)
        
        task = inputs.get('task', '')
        context = inputs.get('context', {})
        
        # 모의 에이전트 실행 과정
        execution_steps = [
            {
                'step': 1,
                'action': 'task_analysis',
                'tool_used': 'reasoning',
                'result': f"작업 분석: '{task}' 수행을 위한 계획 수립"
            },
            {
                'step': 2,
                'action': 'tool_selection',
                'tool_used': 'mcp_router',
                'result': f"사용 가능한 도구: {', '.join(mcp_tools) if mcp_tools else 'web_search, code_exec'}"
            },
            {
                'step': 3,
                'action': 'execution',
                'tool_used': mcp_tools[0] if mcp_tools else 'web_search',
                'result': f"'{task}' 작업을 수행했습니다."
            }
        ]
        
        final_result = f"에이전트 '{agent_name}'가 '{task}' 작업을 성공적으로 완료했습니다."
        
        return {
            'agent_id': agent_id,
            'agent_name': agent_name,
            'task_completed': True,
            'final_result': final_result,
            'execution_steps': execution_steps,
            'mcp_integration': {
                'tools_available': mcp_tools or ['web_search', 'code_execution', 'file_operations'],
                'function_calls_made': len(execution_steps),
                'total_iterations': len(execution_steps)
            },
            'performance_metrics': {
                'execution_time': '2.3s',
                'success_rate': 100,
                'confidence_score': 0.92
            }
        }
        
    except Exception as e:
        logger.error(f"Agent node error: {e}")
        raise Exception(f"Agent node failed: {str(e)}")

def execute_question_classifier_node(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """주제 분류 노드"""
    try:
        question = inputs.get('question', '')
        categories = config.get('categories', [])
        classification_mode = config.get('mode', 'single')  # single, multi
        confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # 기본 카테고리 설정
        default_categories = [
            '기술지원', '제품문의', '결제문의', '일반문의', '불만사항'
        ]
        
        available_categories = categories if categories else default_categories
        
        # 간단한 키워드 기반 분류 (실제로는 ML 모델 사용)
        question_lower = question.lower()
        classification_results = []
        
        # 각 카테고리별 키워드 매핑
        category_keywords = {
            '기술지원': ['오류', '버그', '문제', '안되', '작동', '기술', 'api', '코드'],
            '제품문의': ['기능', '사용법', '어떻게', '방법', '제품', '서비스'],
            '결제문의': ['결제', '요금', '비용', '가격', '청구', '환불'],
            '일반문의': ['정보', '문의', '질문', '궁금', '알려'],
            '불만사항': ['불만', '개선', '불편', '문제점', '건의']
        }
        
        for category in available_categories:
            keywords = category_keywords.get(category, [category.lower()])
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            confidence = min(matches / len(keywords) * 2, 1.0)  # 최대 1.0
            
            if confidence >= confidence_threshold:
                classification_results.append({
                    'category': category,
                    'confidence': confidence,
                    'matching_keywords': [k for k in keywords if k in question_lower]
                })
        
        # 신뢰도순 정렬
        classification_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 분류 모드에 따른 결과 필터링
        if classification_mode == 'single' and classification_results:
            final_results = [classification_results[0]]
        else:
            final_results = classification_results
        
        primary_category = final_results[0]['category'] if final_results else '분류불가'
        
        return {
            'question': question,
            'primary_category': primary_category,
            'all_classifications': final_results,
            'classification_mode': classification_mode,
            'confidence_threshold': confidence_threshold,
            'processing_info': {
                'categories_checked': len(available_categories),
                'matches_found': len(final_results),
                'highest_confidence': final_results[0]['confidence'] if final_results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Question classifier error: {e}")
        raise Exception(f"Question classifier failed: {str(e)}")

# 노드 실행기 매핑
NODE_EXECUTORS = {
    'code': execute_code_node,
    'template_transform': execute_template_transform_node,
    'http_request': execute_http_request_node,
    'variable_assigner': execute_variable_assigner_node,
    'if_else': execute_if_else_node,
    'parameter_extractor': execute_parameter_extractor_node,
    'start': execute_start_node,
    'end': execute_end_node,
    'list_operator': execute_list_operator_node,
    'iteration': execute_iteration_node,
    'knowledge_retrieval': execute_knowledge_retrieval_node,
    'llm': execute_llm_node,
    'agent': execute_agent_node,
    'question_classifier': execute_question_classifier_node,
}

@app.post("/test-node", response_model=NodeTestResponse)
async def test_node(request: NodeTestRequest):
    """노드 테스트 실행"""
    try:
        executor = NODE_EXECUTORS.get(request.node_type)
        
        if not executor:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported node type: {request.node_type}"
            )
        
        result = executor(request.config, request.inputs)
        
        logger.info(f"Node {request.node_type} executed successfully")
        
        return NodeTestResponse(
            status="success",
            node_type=request.node_type,
            result=result
        )
        
    except Exception as e:
        logger.exception(f"Error testing {request.node_type} node")
        return NodeTestResponse(
            status="error",
            node_type=request.node_type,
            result={},
            error=str(e)
        )

@app.get("/node-types")
async def get_supported_node_types():
    """지원하는 노드 타입들"""
    return {
        "supported_types": list(NODE_EXECUTORS.keys()),
        "examples": {
            "code": {
                "config": {
                    "code": "result = inputs['number'] * 2\noutput = {'doubled': result}",
                    "code_language": "python3"
                },
                "inputs": {
                    "number": 5
                }
            },
            "template_transform": {
                "config": {
                    "template": "안녕하세요, {{name}}님! 오늘은 {{date}}입니다."
                },
                "inputs": {
                    "name": "홍길동",
                    "date": "2024-01-15"
                }
            },
            "http_request": {
                "config": {
                    "method": "GET",
                    "url": "https://api.github.com/users/{{username}}",
                    "headers": {}
                },
                "inputs": {
                    "username": "octocat"
                }
            },
            "variable_assigner": {
                "config": {
                    "assignments": [
                        {"variable": "greeting", "value": "안녕하세요"},
                        {"variable": "user_name", "value": "{{name}}"}
                    ]
                },
                "inputs": {
                    "name": "테스터"
                }
            },
            "if_else": {
                "config": {
                    "conditions": [
                        {
                            "variable": "score",
                            "comparison_operator": "gt",
                            "value": 80
                        }
                    ],
                    "logical_operator": "and"
                },
                "inputs": {
                    "score": 90
                }
            },
            "parameter_extractor": {
                "config": {
                    "parameters": [
                        {"name": "name", "type": "string", "required": True},
                        {"name": "age", "type": "number", "required": False}
                    ]
                },
                "inputs": {
                    "text": "안녕하세요, 제 이름은 홍길동이고 나이는 25세입니다."
                }
            },
            "start": {
                "config": {},
                "inputs": {
                    "user_id": "user123",
                    "session_id": "session456"
                }
            },
            "end": {
                "config": {
                    "outputs": {
                        "result": "final_answer",
                        "status": "process_status"
                    }
                },
                "inputs": {
                    "final_answer": "작업이 완료되었습니다.",
                    "process_status": "success"
                }
            },
            "list_operator": {
                "config": {
                    "operation": "append"
                },
                "inputs": {
                    "list": ["apple", "banana"],
                    "item": "orange"
                }
            },
            "iteration": {
                "config": {
                    "max_iterations": 5
                },
                "inputs": {
                    "items": ["task1", "task2", "task3"]
                }
            },
            "knowledge_retrieval": {
                "config": {
                    "dataset_id": "kb_001",
                    "top_k": 3,
                    "output_format": "structured",
                    "search_mode": "hybrid"
                },
                "inputs": {
                    "query": "Python 프로그래밍"
                }
            },
            "llm": {
                "config": {
                    "model": "gpt-4",
                    "prompt_template": "사용자 질문: {{question}}\n컨텍스트: {{context}}\n\n위 내용을 바탕으로 답변해주세요.",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "system_prompt": "당신은 도움이 되는 AI 어시스턴트입니다."
                },
                "inputs": {
                    "question": "Python의 장점은 무엇인가요?",
                    "context": "프로그래밍 언어 비교 자료"
                }
            },
            "agent": {
                "config": {
                    "agent_id": "custom_agent_001",
                    "agent_name": "Data Analysis Agent",
                    "mcp_tools": ["web_search", "code_execution", "file_operations"],
                    "max_iterations": 5
                },
                "inputs": {
                    "task": "데이터 분석 보고서 작성",
                    "context": {"dataset": "sales_data.csv", "period": "2024-Q1"}
                }
            },
            "question_classifier": {
                "config": {
                    "categories": ["기술지원", "제품문의", "결제문의", "일반문의"],
                    "mode": "single",
                    "confidence_threshold": 0.3
                },
                "inputs": {
                    "question": "API가 계속 오류가 나는데 어떻게 해결하나요?"
                }
            }
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Simple Workflow Node Tester",
        "version": "1.0.0",
        "supported_endpoints": [
            "GET / - API 정보",
            "GET /node-types - 지원하는 노드 타입과 예제",
            "POST /test-node - 노드 테스트 실행"
        ]
    }