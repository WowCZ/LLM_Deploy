import os
from typing import List, Union

class LLMAPI:
    def __init__(self, model_name: str, 
                 model_path: Union[str, dict]=None, 
                 model_version: str='default', 
                 adapter_name: str=None, 
                 adapter_path: str=None):
        
        self.model_name = model_name
        if model_path and type(model_path) is dict:
            model_path = model_path[model_version]
        
            assert type(model_path) is str, f'type of model_path is {type(model_path)} rather than string.'

        self.model_path = model_path
        self.adapter_name = adapter_name
        self.adapter_path = adapter_path
        self.model_version = model_version
        self.supported_types = ['generate']

        if model_path and not os.path.exists(model_path):
            self._download_llm(model_name, model_path)

        if adapter_path and not os.path.exists(adapter_path):
            self._download_adapter(adapter_name, adapter_path)
        
        self.model, self.tokenizer = self._initialize_llm()
    
    def _download_llm(self, model_name: str, model_path: str):
        # if model_path is not existed, download the model parameters into the custom path
        pass

    def _download_adapter(self, adapter_name: str, adapter_path: str):
        # if model_path is not existed, download the model parameters into the custom path
        pass

    def _initialize_llm(self):
        # initialize the llm with the downloaded model_path
        return None, None
    
    def generate(self, instance: Union[str, list]) -> List[str]:
        # generate the output with the initialized model
        return []
    
    def score(self, instance: Union[str, list]) -> list:
        # caculate the probability of the candidate answer with the initialized model
        return [0.0]