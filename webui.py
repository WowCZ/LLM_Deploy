import argparse
from cli import deploy_arena, deploy_chat

def arena(args):
    deploy_arena('offline', 
                args.port)
    
def chat(args):
    deploy_chat(args.url, 
                args.port)

parser = argparse.ArgumentParser(description='WebUI based on Gradio')
subparsers = parser.add_subparsers(help='WebUI')

arena_parser = subparsers.add_parser(name='arena', help='Arena WebUI')
arena_parser.add_argument('--port', type=int, default=8888, help='deploy port')
arena_parser.set_defaults(func=arena)


chat_parser = subparsers.add_parser(name='chat', help='Chat WebUI')
chat_parser.add_argument('--url', type=str, default='http://10.140.0.31:7437/generate', help='language model api')
chat_parser.add_argument('--port', type=int, default=8888, help='deploy port')
chat_parser.set_defaults(func=chat)

args = parser.parse_args()
args.func(args)
