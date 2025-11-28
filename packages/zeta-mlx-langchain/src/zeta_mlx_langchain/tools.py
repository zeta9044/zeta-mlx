"""LangChain Tool Calling 지원"""
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from langchain_core.tools import BaseTool, StructuredTool
from zeta_mlx_core import Message, ChatRequest, GenerationParams
from zeta_mlx_inference import InferenceEngine


def create_tool_prompt(tools: List[BaseTool], query: str) -> str:
    """도구 설명을 포함한 프롬프트 생성

    Note: MLX LLM은 네이티브 tool calling을 지원하지 않으므로
    프롬프트 기반 도구 호출을 사용합니다.
    """
    tool_descriptions = []
    for tool in tools:
        desc = f"- {tool.name}: {tool.description}"
        if hasattr(tool, "args_schema") and tool.args_schema:
            schema = tool.args_schema.model_json_schema()
            if "properties" in schema:
                params = ", ".join(schema["properties"].keys())
                desc += f" (parameters: {params})"
        tool_descriptions.append(desc)

    tools_text = "\n".join(tool_descriptions)

    return f"""You have access to the following tools:

{tools_text}

To use a tool, respond in this format:
TOOL: <tool_name>
ARGS: <json_arguments>

If no tool is needed, just respond normally.

User query: {query}"""


def parse_tool_response(response: str) -> Optional[tuple[str, dict]]:
    """도구 호출 응답 파싱

    Returns:
        (tool_name, arguments) 또는 None (도구 호출이 없는 경우)
    """
    import json

    lines = response.strip().split("\n")
    tool_name = None
    args_str = None

    for i, line in enumerate(lines):
        if line.startswith("TOOL:"):
            tool_name = line[5:].strip()
        elif line.startswith("ARGS:"):
            args_str = line[5:].strip()
            # 다음 줄까지 JSON이 이어질 수 있음
            for j in range(i + 1, len(lines)):
                if not lines[j].startswith("TOOL:") and not lines[j].startswith("ARGS:"):
                    args_str += lines[j]
                else:
                    break

    if tool_name and args_str:
        try:
            args = json.loads(args_str)
            return (tool_name, args)
        except json.JSONDecodeError:
            return None

    return None


class ToolExecutor:
    """도구 실행기

    LangChain 도구들을 MLX LLM과 함께 사용할 수 있게 해주는 실행기
    """

    def __init__(
        self,
        engine: InferenceEngine,
        tools: List[BaseTool],
        max_iterations: int = 3,
    ) -> None:
        self._engine = engine
        self._tools = {tool.name: tool for tool in tools}
        self._tools_list = tools
        self._max_iterations = max_iterations

    def run(self, query: str, model_name: str = "") -> str:
        """도구를 사용하여 쿼리 처리"""
        from zeta_mlx_core import Success, Failure

        messages = []
        current_query = query

        for _ in range(self._max_iterations):
            # 도구 프롬프트 생성
            prompt = create_tool_prompt(self._tools_list, current_query)

            messages.append(Message(role="user", content=prompt))

            request = ChatRequest(
                model=model_name,
                messages=messages,
                params=GenerationParams(),
                stream=False,
            )

            result = self._engine.generate(request)

            if isinstance(result, Failure):
                return f"Error: {result.error}"

            response_text = result.value.text
            messages.append(Message(role="assistant", content=response_text))

            # 도구 호출 파싱
            tool_call = parse_tool_response(response_text)

            if tool_call is None:
                # 도구 호출 없음, 최종 응답 반환
                return response_text

            tool_name, args = tool_call

            if tool_name not in self._tools:
                return f"Error: Unknown tool '{tool_name}'"

            # 도구 실행
            tool = self._tools[tool_name]
            try:
                tool_result = tool.invoke(args)
                current_query = f"Tool '{tool_name}' returned: {tool_result}\n\nBased on this result, answer the original question: {query}"
            except Exception as e:
                current_query = f"Tool '{tool_name}' failed with error: {str(e)}\n\nPlease try a different approach to answer: {query}"

        return "Max iterations reached. Please simplify your query."


def create_tool_executor(
    engine: InferenceEngine,
    tools: List[BaseTool],
    max_iterations: int = 3,
) -> ToolExecutor:
    """도구 실행기 생성"""
    return ToolExecutor(engine, tools, max_iterations)
