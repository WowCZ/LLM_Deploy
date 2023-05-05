import logging

def get_logger(name, level='DEBUG'):
    logging_level = eval(f'logging.{level}')
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

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
           'MOSSAPI',
           'get_logger']
