import sys
sys.path.append('../assets')
sys.path.append('../analysis')

from .backend import record_as_json
from .arena import arena_two_model
from .chat import chat_with_api