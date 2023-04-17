import os
from typing import List, Union

class LLMAPI:
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.model_path = model_path
        if model_path and not os.path.exists(model_path):
            self._download_llm(model_name, model_path)
        
        self.model, self.tokenizer = self._initialize_llm()
    
    def _download_llm(self, model_name: str, model_path: str) -> bool:
        # if model_path is not existed, download the model parameters into the custom path
        return True

    def _initialize_llm(self):
        # initialize the llm with the downloaded model_path
        return None, None
    
    def generate(self, instance: Union[str, list]) -> List[str]:
        # generate the output with the initialized model
        return []
    
    def score(self, instance: Union[str, list]) -> float:
        # caculate the probability of the candidate answer with the initialized model
        return 0.0