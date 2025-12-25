import random
from openai import OpenAI

class LLMAgent(object):

    def __init__(self, llm, model_config, prompt_function, out_parse_function, action_convert_function):
        self.use_raw = True
        self.llm = llm
        self.model_config = model_config
        self.client = OpenAI(
            base_url=self.model_config['call_func']['base_url'],
            api_key=self.model_config['call_func']['api_key'],
        )
        self.prompt_function = prompt_function
        self.out_parse_function = out_parse_function
        self.action_convert_function = action_convert_function
        self.request_count = 0
        self.correct_count = 0

    def step(self, state):
        if len(state['raw_legal_actions']) == 1:
            return state['raw_legal_actions'][0]
        
        prompt = self.prompt_function(state)
        output = self.llm(prompt, client=self.client, model_config=self.model_config)
        action = self.out_parse_function(output)

        state['raw_obs']['output'] = output
        self.request_count += 1

        if not action or action not in state['raw_legal_actions']:
            # action = state['raw_legal_actions'][0]
            # random select an action
            action = random.choice(state['raw_legal_actions'])
        else:
            self.correct_count += 1
        # print(action)
        return self.action_convert_function(action)

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []
