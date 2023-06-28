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

logger = get_logger(__name__, level='INFO')


api_name_map = {
    'gpt4': 'gpt-4',
    'alpaca': 'Aplaca-LoRA-7B',
    'belle': 'BELLE-7B',
    'vicuna': 'Vicuna-7B',
    'turbo': 'gpt-3.5-turbo',
    'chatglm': 'ChatGLM-6B',
    'baichuan-vicuna': 'BaiChuan-Chinese-Vicuna-7B',
    'bloom': 'BLOOM-7B1',
    'chinese-vicuna': 'Chinese-Vicuna-7B',
    'davinci': 'text-davinci-003',
    'llama': 'LLaMA-7B',
    'bloomz-mt': 'BLOOMZ-MT-7B1',
    'chinese-alpaca': 'Chinese-Alpaca-LoRA-7B',
    'moss': 'MOSS-moon-003-sft-16B',
    'vicuna-13b': 'Vicuna-13B',
    'sensechat': 'SenseChat',
    'baichuan': 'BaiChuan-7B',
}

ability_name_map = {
    '长文理解': '长文阅读能力',
    '言外之意': '言外之意理解能力',
    '创意表达': '创意表达能力',
    '思辨能力': '思辨能力',
    '长文表达': '强逻辑长文表达能力',
    '古诗词鉴赏': '古诗词鉴赏能力',
    '共情对话': '共情对话能力',
    '安全能力': '安全交互能力',
    '幽默理解': '幽默理解能力',
    '常识推理': '常识推理解释能力'
}

ability_en_zh_map = {
    'reading': '长文阅读能力',
    'hinting': '言外之意理解能力',
    'story': '创意表达能力',
    'philosophical': '思辨能力',
    'writing': '强逻辑长文表达能力',
    'poetry': '古诗词鉴赏能力',
    'empathy': '共情对话能力',
    'safety': '安全交互能力',
    'humor': '幽默理解能力',
    'reasoning': '常识推理解释能力',
    'overall': '综合能力'
}

model_download_path = '/mnt/lustre/chenzhi/workspace/LLM/models'

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
from .gpt4 import GPT4API
from .chinese_alpaca import ChineseAlpacaAPI
from .stablelm import StablelmAPI
from .chinese_vicuna import ChineseVicunaAPI
from .belle import BELLEAPI
from .cpm import CPMAPI
from .sensechat import SenseChatAPI
from .baichuan import BaiChuanAPI

__all__ = ['AlpacaAPI',
           'BaiChuanAPI',
           'BELLEAPI', 
           'BloomAPI', 
           'ChatGLMAPI', 
           'ChineseAlpacaAPI',
           'ChineseVicunaAPI',
           'CPMAPI',
           'DavinciAPI', 
           'GPT4API',
           'LLaMAAPI', 
           'MOSSAPI',
           'StablelmAPI',
           'T5API', 
           'TurboAPI', 
           'VicunaAPI',
           'SenseChatAPI']

logger.info(f'Now we support to deploy {len(__all__)} large language models as APIs, listed as {__all__}')
