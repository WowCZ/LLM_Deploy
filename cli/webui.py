from webui import arena_two_model

def deploy_webui(type: str, port: int):
    if type == 'arena':
        arena_two_model()