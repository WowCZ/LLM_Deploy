from webui import arena_two_model, chat_with_api

def deploy_arena(type: str, port: int):
    if type == 'offline':
        arena_two_model(port)

def deploy_chat(url: str, port: int):
    chat_with_api(url, port)