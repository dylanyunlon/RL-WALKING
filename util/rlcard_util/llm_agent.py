import random
import numpy as np
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
            # For single action, return it directly (already in correct format)
            action = state['raw_legal_actions'][0]
            # If action is already int (gin-rummy), convert it properly
            if isinstance(action, int):
                return self.action_convert_function(action)
            return action
        
        prompt = self.prompt_function(state)
        output = self.llm(prompt, client=self.client, model_config=self.model_config)
        action = self.out_parse_function(output)

        # Fix: Only store output if raw_obs is a dict, not numpy array
        raw_obs = state.get('raw_obs')
        if isinstance(raw_obs, dict):
            raw_obs['output'] = output
        # For numpy arrays (like gin-rummy), we skip storing output in raw_obs
        
        self.request_count += 1

        # Check if action is valid
        # Need to handle both string actions and int actions (gin-rummy)
        raw_legal_actions = state['raw_legal_actions']
        
        # Determine if legal_actions are integers (gin-rummy) or strings (other games)
        legal_actions_are_ints = raw_legal_actions and isinstance(raw_legal_actions[0], int)
        
        if legal_actions_are_ints:
            # For gin-rummy: raw_legal_actions are integers
            # action from LLM is a string, need to convert and check
            converted_action = self.action_convert_function(action) if action else None
            
            # Check if converted action is valid
            action_valid = False
            if converted_action is not None:
                # Get action ID from converted action
                if hasattr(converted_action, 'action_id'):
                    action_id = converted_action.action_id
                    action_valid = action_id in raw_legal_actions
                elif isinstance(converted_action, int):
                    action_valid = converted_action in raw_legal_actions
            
            if not action_valid:
                # Random fallback: pick from legal actions and convert
                random_action_id = random.choice(raw_legal_actions)
                return self.action_convert_function(random_action_id)
            else:
                self.correct_count += 1
                return converted_action
        else:
            # For other games: raw_legal_actions are strings
            if not action or action not in raw_legal_actions:
                action = random.choice(raw_legal_actions)
            else:
                self.correct_count += 1
            return self.action_convert_function(action)

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []