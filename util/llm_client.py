import os, json
import requests
import re
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception

temperature = os.getenv('API_TEMP', 0)

class ModelCallError(Exception):
    """可重试的模型调用错误"""
    pass

class ContextTooLongError(Exception):
    """上下文超长错误 - 不应重试，直接跳过"""
    pass

def is_retryable_error(exception):
    """判断是否应该重试该异常"""
    # 上下文超长错误不应重试
    if isinstance(exception, ContextTooLongError):
        return False
    # 其他 ModelCallError 可以重试
    if isinstance(exception, ModelCallError):
        return True
    return False

def openai_api_function(prompt, history=None, system=None, client=None, model_config=None):
    message = []
    if system:
        message.append({
            "role": "system",
            "content": system
        })
    
    if history:
        for chat in history:
            message.append({
                "role": "user",
                "content": chat[0]
            })
            message.append({
                "role": "assistant",
                "content": chat[1]
            })
    
    message.append({
        "role": "user",
        "content": prompt
    })

    resp = client.with_options(max_retries=5).chat.completions.create(
        messages=message,
        model=model_config['model_type'],
        temperature=temperature,
        stream=False,
        max_tokens=1024
    )
    output = resp.choices[0].message.content

    return output

@retry(
    stop=stop_after_attempt(2), 
    wait=wait_fixed(10), 
    retry=retry_if_exception(is_retryable_error)
)
def llm_function(prompt, history=None, system=None, client=None, model_config=None):
    try:
        output = openai_api_function(prompt, history, system, client, model_config)
        return output
    except Exception as e:
        error_str = str(e)
        # 检查是否是上下文超长错误 - 这些错误不应该重试
        if any(keyword in error_str.lower() for keyword in [
            "maximum context length",
            "context length",
            "too many tokens",
            "token limit",
            "reduce the length"
        ]):
            print(f"[SKIP] Context too long, will use random action: {error_str[:150]}...")
            raise ContextTooLongError(f"Context too long: {error_str[:150]}")
        
        # 检查 400 Bad Request（通常是输入问题）
        if "400" in error_str and "BadRequestError" in error_str:
            # 进一步检查是否是上下文问题
            if "token" in error_str.lower() or "length" in error_str.lower():
                print(f"[SKIP] Bad request (context issue), will use random action: {error_str[:150]}...")
                raise ContextTooLongError(f"Bad request: {error_str[:150]}")
        
        # 其他错误，可以重试
        raise ModelCallError(f"Error calling model: {str(e)}")