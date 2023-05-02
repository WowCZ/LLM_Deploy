from . import base_api
from .base_api import LLMAPI
from .chatglm import ChatGLMAPI
from .davinci import DavinciAPI
from .t5 import T5API
from .turbo import TurboAPI
from .bloom import BloomAPI
from .llama import LLaMAAPI
from .vicuna import VicunaAPI
from .alpaca import AlpacaAPI
from .moss import MOSSAPI
# from .gpt4 import GPT4API

__all__ = ['LLMAPI', 
           'ChatGLMAPI', 
           'T5API', 
           'DavinciAPI', 
           'TurboAPI', 
           'BloomAPI', 
           'LLaMAAPI', 
           'VicunaAPI',
           'AlpacaAPI',
           'MOSSAPI']