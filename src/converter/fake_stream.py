from typing import Any, Dict, List, Tuple
import json
from src.converter.utils import extract_content_and_reasoning
from log import log
from src.converter.openai2gemini import _convert_usage_metadata, extract_tool_calls_from_parts

# ... (keep existing imports)

# ... (keep existing imports)

def safe_get_nested(obj: Any, *keys: str, default: Any = None) -> Any:
    """安全获取嵌套字典值
    
    Args:
        obj: 字典对象
        *keys: 嵌套键路径
        default: 默认值
    
    Returns:
        获取到的值或默认值
    """
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key, default)
        if obj is default:
            return default
    return obj

def parse_response_for_fake_stream(response_data: Dict[str, Any]) -> tuple:
    """从完整响应中提取内容、推理内容和工具调用(用于假流式)

    Args:
        response_data: Gemini API 响应数据

    Returns:
        (content, reasoning_content, finish_reason, images, tool_calls): 元组
    """
    import json

    # 处理GeminiCLI的response包装格式
    if "response" in response_data and "candidates" not in response_data:
        log.debug(f"[FAKE_STREAM] Unwrapping response field")
        response_data = response_data["response"]

    candidates = response_data.get("candidates", [])
    log.debug(f"[FAKE_STREAM] Found {len(candidates)} candidates")
    if not candidates:
        return "", "", "STOP", [], []

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "STOP")
    parts = safe_get_nested(candidate, "content", "parts", default=[])
    log.debug(f"[FAKE_STREAM] Extracted {len(parts)} parts: {json.dumps(parts, ensure_ascii=False)}")
    content, reasoning_content, images = extract_content_and_reasoning(parts)
    
    # 提取工具调用
    # is_streaming=True 会给每个 tool_call 添加 index 字段，这对流式响应很重要
    tool_calls, _ = extract_tool_calls_from_parts(parts, is_streaming=True)
    
    log.debug(f"[FAKE_STREAM] Content length: {len(content)}, Reasoning length: {len(reasoning_content)}, Images count: {len(images)}, Tool calls: {len(tool_calls)}")

    return content, reasoning_content, finish_reason, images, tool_calls

def extract_fake_stream_content(response: Any) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    """
    从 Gemini 非流式响应中提取内容，用于假流式处理
    
    Args:
        response: Gemini API 响应对象
    
    Returns:
        (content, reasoning_content, usage, tool_calls) 元组
    """
    from src.converter.utils import extract_content_and_reasoning
    
    # 解析响应体
    if hasattr(response, "body"):
        body_str = (
            response.body.decode()
            if isinstance(response.body, bytes)
            else str(response.body)
        )
    elif hasattr(response, "content"):
        body_str = (
            response.content.decode()
            if isinstance(response.content, bytes)
            else str(response.content)
        )
    else:
        body_str = str(response)

    try:
        response_data = json.loads(body_str)

        # GeminiCLI 返回的格式是 {"response": {...}, "traceId": "..."}
        # 需要先提取 response 字段
        if "response" in response_data:
            gemini_response = response_data["response"]
        else:
            gemini_response = response_data

        # 从Gemini响应中提取内容，使用思维链分离逻辑
        content = ""
        reasoning_content = ""
        images = []
        tool_calls = []
        
        if "candidates" in gemini_response and gemini_response["candidates"]:
            # Gemini格式响应 - 使用思维链分离
            candidate = gemini_response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                content, reasoning_content, images = extract_content_and_reasoning(parts)
                # 提取工具调用
                tool_calls, _ = extract_tool_calls_from_parts(parts, is_streaming=True)
        elif "choices" in gemini_response and gemini_response["choices"]:
            # OpenAI格式响应
            content = gemini_response["choices"][0].get("message", {}).get("content", "")
            # OpenAI格式暂不需额外处理工具调用，因为通常来自Gemini格式

        # 如果没有正常内容但有思维内容，给出警告
        if not content and reasoning_content and not tool_calls:
            log.warning("Fake stream response contains only thinking content")
            content = "[模型正在思考中，请稍后再试或重新提问]"
        
        # 如果完全没有内容也没有工具调用，提供默认回复
        if not content and not tool_calls:
            log.warning(f"No content found in response: {gemini_response}")
            content = "[响应为空，请重新尝试]"

        # 转换usageMetadata为OpenAI格式
        usage = _convert_usage_metadata(gemini_response.get("usageMetadata"))
        
        return content, reasoning_content, usage, tool_calls

    except json.JSONDecodeError:
        # 如果不是JSON，直接返回原始文本
        return body_str, "", None, []


def _build_candidate(parts: List[Dict[str, Any]], finish_reason: str = "STOP") -> Dict[str, Any]:
    """构建标准候选响应结构
    
    Args:
        parts: parts 列表
        finish_reason: 结束原因
    
    Returns:
        候选响应字典
    """
    return {
        "candidates": [{
            "content": {"parts": parts, "role": "model"},
            "finishReason": finish_reason,
            "index": 0,
        }]
    }

def create_openai_heartbeat_chunk() -> Dict[str, Any]:
    """
    创建 OpenAI 格式的心跳块（用于假流式）
    
    Returns:
        心跳响应块字典
    """
    return {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ]
    }

def build_gemini_fake_stream_chunks(content: str, reasoning_content: str, finish_reason: str, images: List[Dict[str, Any]] = None, chunk_size: int = 50) -> List[Dict[str, Any]]:
    """构建假流式响应的数据块

    Args:
        content: 主要内容
        reasoning_content: 推理内容
        finish_reason: 结束原因
        images: 图片数据列表（可选）
        chunk_size: 每个chunk的字符数（默认50）

    Returns:
        响应数据块列表
    """
    if images is None:
        images = []

    log.debug(f"[build_gemini_fake_stream_chunks] Input - content: {repr(content)}, reasoning: {repr(reasoning_content)}, finish_reason: {finish_reason}, images count: {len(images)}")
    chunks = []

    # 如果没有正常内容但有思维内容,提供默认回复
    if not content:
        default_text = "[模型正在思考中,请稍后再试或重新提问]" if reasoning_content else "[响应为空,请重新尝试]"
        return [_build_candidate([{"text": default_text}], finish_reason)]

    # 分块发送主要内容
    first_chunk = True
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i + chunk_size]
        is_last_chunk = (i + chunk_size >= len(content)) and not reasoning_content
        chunk_finish_reason = finish_reason if is_last_chunk else None

        # 如果是第一个chunk且有图片，将图片包含在parts中
        parts = []
        if first_chunk and images:
            # 在Gemini格式中，需要将image_url格式转换为inlineData格式
            for img in images:
                if img.get("type") == "image_url":
                    url = img.get("image_url", {}).get("url", "")
                    # 解析 data URL: data:{mime_type};base64,{data}
                    if url.startswith("data:"):
                        parts_of_url = url.split(";base64,")
                        if len(parts_of_url) == 2:
                            mime_type = parts_of_url[0].replace("data:", "")
                            base64_data = parts_of_url[1]
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64_data
                                }
                            })
            first_chunk = False

        parts.append({"text": chunk_text})
        chunk_data = _build_candidate(parts, chunk_finish_reason)
        log.debug(f"[build_gemini_fake_stream_chunks] Generated chunk: {chunk_data}")
        chunks.append(chunk_data)

    # 如果有推理内容，分块发送
    if reasoning_content:
        for i in range(0, len(reasoning_content), chunk_size):
            chunk_text = reasoning_content[i:i + chunk_size]
            is_last_chunk = i + chunk_size >= len(reasoning_content)
            chunk_finish_reason = finish_reason if is_last_chunk else None
            chunks.append(_build_candidate([{"text": chunk_text, "thought": True}], chunk_finish_reason))

    log.debug(f"[build_gemini_fake_stream_chunks] Total chunks generated: {len(chunks)}")
    return chunks


def create_gemini_heartbeat_chunk() -> Dict[str, Any]:
    """创建 Gemini 格式的心跳数据块

    Returns:
        心跳数据块
    """
    chunk = _build_candidate([{"text": ""}])
    chunk["candidates"][0]["finishReason"] = None
    return chunk

def build_openai_fake_stream_chunks(content: str, reasoning_content: str, finish_reason: str, model: str, images: List[Dict[str, Any]] = None, chunk_size: int = 50, tool_calls: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """构建 OpenAI 格式的假流式响应数据块
    
    Args:
        content: 主要内容
        reasoning_content: 推理内容
        finish_reason: 结束原因
        model: 模型名称
        images: 图片数据
        chunk_size: 大小
        tool_calls: 工具调用列表
    """
    import time
    import uuid

    if images is None:
        images = []
    if tool_calls is None:
        tool_calls = []

    log.debug(f"[build_openai_fake_stream_chunks] Input - content: {len(content)} chars, reasoning: {len(reasoning_content)} chars, finish_reason: {finish_reason}, images: {len(images)}, tool_calls: {len(tool_calls)}")
    chunks = []
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # 映射 Gemini finish_reason 到 OpenAI 格式
    openai_finish_reason = None
    if tool_calls:
        openai_finish_reason = "tool_calls"
    elif finish_reason == "STOP":
        openai_finish_reason = "stop"
    elif finish_reason == "MAX_TOKENS":
        openai_finish_reason = "length"
    elif finish_reason in ["SAFETY", "RECITATION"]:
        openai_finish_reason = "content_filter"

    # 1. 处理工具调用
    if tool_calls:
        for tc in tool_calls:
            index = tc.get("index", 0)
            func = tc.get("function", {})
            
            # 第一帧：ID, Type, Name
            delta_header = {
                "tool_calls": [{
                    "index": index,
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "arguments": ""
                    }
                }]
            }
            chunks.append({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta_header,
                    "finish_reason": None
                }]
            })
            
            # 第二帧：Arguments (一次性发送)
            delta_args = {
                "tool_calls": [{
                    "index": index,
                    "function": {
                        "arguments": func.get("arguments", "{}")
                    }
                }]
            }
            chunks.append({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta_args,
                    "finish_reason": None # 最后一个tool call结束时才设置finish_reason? OpenAI通常是流式
                }]
            })
            
        # 如果只有工具调用没有内容，发送结束帧
        if not content and not reasoning_content:
             chunks[-1]["choices"][0]["finish_reason"] = openai_finish_reason
             return chunks

    # 如果没有正常内容但有思维内容，提供默认回复 (仅当没有工具调用时)
    if not content and not tool_calls:
        default_text = "[模型正在思考中，请稍后再试或重新提问]" if reasoning_content else "[响应为空，请重新尝试]"
        return [{
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": default_text},
                "finish_reason": openai_finish_reason,
            }]
        }]

    # 分块发送主要内容 (同前)
    # ... (content handling)
    first_chunk = True
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i + chunk_size]
        is_last_chunk = (i + chunk_size >= len(content)) and not reasoning_content
        # 如果有工具调用，文本结束不一定是整个response结束
        chunk_finish = openai_finish_reason if is_last_chunk else None

        delta_content = {}
        if first_chunk and images:
             delta_content["content"] = images + [{"type": "text", "text": chunk_text}]
             first_chunk = False
        else:
             delta_content["content"] = chunk_text

        chunk_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta_content,
                "finish_reason": chunk_finish,
            }]
        }
        chunks.append(chunk_data)

    # ... (reasoning handling)
    if reasoning_content:
        for i in range(0, len(reasoning_content), chunk_size):
            chunk_text = reasoning_content[i:i + chunk_size]
            is_last_chunk = i + chunk_size >= len(reasoning_content)
            chunk_finish = openai_finish_reason if is_last_chunk else None

            chunks.append({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": chunk_text},
                    "finish_reason": chunk_finish,
                }]
            })

    return chunks


def create_anthropic_heartbeat_chunk() -> Dict[str, Any]:
    """
    创建 Anthropic 格式的心跳块（用于假流式）

    Returns:
        心跳响应块字典
    """
    return {
        "type": "ping"
    }


def build_anthropic_fake_stream_chunks(content: str, reasoning_content: str, finish_reason: str, model: str, images: List[Dict[str, Any]] = None, chunk_size: int = 50) -> List[Dict[str, Any]]:
    """构建 Anthropic 格式的假流式响应数据块

    Args:
        content: 主要内容
        reasoning_content: 推理内容（thinking content）
        finish_reason: 结束原因（如 "STOP", "MAX_TOKENS"）
        model: 模型名称
        images: 图片数据列表（可选）
        chunk_size: 每个chunk的字符数（默认50）

    Returns:
        Anthropic SSE 格式的响应数据块列表
    """
    import uuid

    if images is None:
        images = []

    log.debug(f"[build_anthropic_fake_stream_chunks] Input - content: {repr(content)}, reasoning: {repr(reasoning_content)}, finish_reason: {finish_reason}, images count: {len(images)}")
    chunks = []
    message_id = f"msg_{uuid.uuid4().hex}"

    # 映射 Gemini finish_reason 到 Anthropic 格式
    anthropic_stop_reason = "end_turn"
    if finish_reason == "MAX_TOKENS":
        anthropic_stop_reason = "max_tokens"
    elif finish_reason in ["SAFETY", "RECITATION"]:
        anthropic_stop_reason = "end_turn"

    # 1. 发送 message_start 事件
    chunks.append({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    })

    # 如果没有正常内容但有思维内容，提供默认回复
    if not content:
        default_text = "[模型正在思考中，请稍后再试或重新提问]" if reasoning_content else "[响应为空，请重新尝试]"

        # content_block_start
        chunks.append({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })

        # content_block_delta
        chunks.append({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": default_text}
        })

        # content_block_stop
        chunks.append({
            "type": "content_block_stop",
            "index": 0
        })

        # message_delta
        chunks.append({
            "type": "message_delta",
            "delta": {"stop_reason": anthropic_stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0}
        })

        # message_stop
        chunks.append({
            "type": "message_stop"
        })

        return chunks

    block_index = 0

    # 2. 如果有推理内容，先发送 thinking 块
    if reasoning_content:
        # thinking content_block_start
        chunks.append({
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "thinking", "thinking": ""}
        })

        # 分块发送推理内容
        for i in range(0, len(reasoning_content), chunk_size):
            chunk_text = reasoning_content[i:i + chunk_size]
            chunks.append({
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": chunk_text}
            })

        # thinking content_block_stop
        chunks.append({
            "type": "content_block_stop",
            "index": block_index
        })

        block_index += 1

    # 3. 如果有图片，发送图片块
    if images:
        for img in images:
            if img.get("type") == "image_url":
                url = img.get("image_url", {}).get("url", "")
                # 解析 data URL: data:{mime_type};base64,{data}
                if url.startswith("data:"):
                    parts_of_url = url.split(";base64,")
                    if len(parts_of_url) == 2:
                        mime_type = parts_of_url[0].replace("data:", "")
                        base64_data = parts_of_url[1]

                        # image content_block_start
                        chunks.append({
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_data
                                }
                            }
                        })

                        # image content_block_stop
                        chunks.append({
                            "type": "content_block_stop",
                            "index": block_index
                        })

                        block_index += 1

    # 4. 发送主要内容（text 块）
    # text content_block_start
    chunks.append({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {"type": "text", "text": ""}
    })

    # 分块发送主要内容
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i + chunk_size]
        chunks.append({
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "text_delta", "text": chunk_text}
        })

    # text content_block_stop
    chunks.append({
        "type": "content_block_stop",
        "index": block_index
    })

    # 5. 发送 message_delta
    chunks.append({
        "type": "message_delta",
        "delta": {"stop_reason": anthropic_stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": len(content) + len(reasoning_content)}
    })

    # 6. 发送 message_stop
    chunks.append({
        "type": "message_stop"
    })

    log.debug(f"[build_anthropic_fake_stream_chunks] Total chunks generated: {len(chunks)}")
    return chunks