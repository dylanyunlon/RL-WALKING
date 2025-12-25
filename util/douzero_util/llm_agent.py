import random
import sys
from openai import OpenAI

# 导入自定义异常（用于类型检查）
# 注意：由于可能的循环导入问题，我们在 except 中用字符串匹配
# from util.llm_client import ContextTooLongError

class LLMAgent(object):

    def __init__(self, llm, model_config, prompt_function, out_parse_function):
        self.use_raw = True
        self.llm = llm
        self.model_config = model_config
        self.client = OpenAI(
            base_url=self.model_config['call_func']['base_url'],
            api_key=self.model_config['call_func']['api_key'],
        )
        self.prompt_function = prompt_function
        self.out_parse_function = out_parse_function
        self.request_count = 0
        self.correct_count = 0
        self.context_too_long_count = 0  # 记录上下文超长次数

    def step(self, state):
        if len(state.legal_actions) == 1:
            return state.legal_actions[0]
        
        prompt = self.prompt_function(state)
        
        try:
            output = self.llm(prompt, client=self.client, model_config=self.model_config)
            action = self.out_parse_function(output)
            state.output = output
            self.request_count += 1
            
            if action is None or action not in state.legal_actions:
                # random select an action
                action = random.choice(state.legal_actions)
            else:
                self.correct_count += 1
                
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            # 检查是否是上下文超长错误
            is_context_error = (
                error_type == "ContextTooLongError" or
                "ContextTooLongError" in error_str or
                "context too long" in error_str.lower() or
                "maximum context length" in error_str.lower() or
                "too many tokens" in error_str.lower()
            )
            
            if is_context_error:
                self.context_too_long_count += 1
                self.request_count += 1
                print(f"[SKIP] Context too long (total: {self.context_too_long_count}), using random action")
                # 上下文超长时，随机选择一个合法动作
                action = random.choice(state.legal_actions)
                state.output = f"[CONTEXT_TOO_LONG] Random action selected: {action}"
            else:
                # 其他错误，重新抛出
                print(f"[ERROR] Unexpected error: {error_type}: {error_str[:200]}")
                raise e
        
        return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []
    
    # for DouZero
    def act(self, state):
        return self.step(state)