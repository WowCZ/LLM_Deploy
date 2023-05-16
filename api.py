import argparse
from cli import api_server, api_client, api_simulator

def server(args):
    gen_config = {
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'top_p': args.top_p,
        'num_return': args.num_return,
        'do_sample': args.do_sample,
        'seed': args.seed
    }

    api_server(args.api, args.wrapper, args.url_save_path, gen_config)

def client(args):
    api_client(args.model_name, 
               args.batch_size, 
               args.max_length, 
               args.url_path, 
               args.evaluation_tasks,
               args.inference_path)
    
def simulator(args):
    api_simulator(args.simulate_task,
                  args.model_name,
                  args.port,
                  args.wrapper,
                  args.url_path,
                  args.inference_path)

parser = argparse.ArgumentParser(description='API Configuration')
subparsers = parser.add_subparsers(help='API Operations')

server_parser = subparsers.add_parser(name='server', help='Deploy api servers')
server_parser.add_argument('--api', type=str, default='ChatGLMAPI', help='Supported API: [ChatGLMAPI, T5API, DavinciAPI, TurboAPI, BloomAPI]')
server_parser.add_argument('--wrapper', type=str, default='Flask', help='Supported Server: [Flask, FastAPI]')
server_parser.add_argument('--url_save_path', type=str, default='copywriting/urls', help='the path of the saving urls.')
server_parser.add_argument('--temperature', type=float, default=0.0, help='generation configuration: temperature')
server_parser.add_argument('--max_new_tokens', type=int, default=1024, help='generation configuration: max_new_tokens')
server_parser.add_argument('--top_p', type=float, default=1.0, help='generation configuration: top_p')
server_parser.add_argument('--num_return', type=int, default=1, help='generation configuration: num_return')
server_parser.add_argument('--do_sample', type=bool, default=False, help='generation configuration: do_sample')
server_parser.add_argument('--seed', type=int, default=42, help='generation configuration: seed')
server_parser.set_defaults(func=server)

client_parser =  subparsers.add_parser(name='client', help='Response with the deployed api servers')
client_parser.add_argument('--model_name', type=str, default='alpaca', help='Supported model names: [alpaca, bloom, chatglm, davinci, llama, moss, t5, turbo, vicuna]')
client_parser.add_argument('--batch_size', type=int, default=1, help='Batch size for each api model.')
client_parser.add_argument('--max_length', type=int, default=1500, help='the maximum length of the prompts.')
client_parser.add_argument('--url_path', type=str, default='copywriting/urls', help='the path of the saved urls.')
client_parser.add_argument('--evaluation_tasks', nargs='+', type=str, default=['empathy', 'hinting', 'humor', 'philosophical', 'poetry', 'reading', 'reasoning', 'safety', 'story', 'writing'], help='Human evaluation tasks')
client_parser.add_argument('--inference_path', type=str, default='copywriting/data', help='the path of the saved urls.')
client_parser.set_defaults(func=client)

simulator_parser =  subparsers.add_parser(name='simulator', help='Simulator')
simulator_parser.add_argument('--simulate_task', type=str, default='empathy', help='human evaluation task')
simulator_parser.add_argument('--model_name', type=str, default='chatglm', help='model name')
simulator_parser.add_argument('--port', type=int, default=6565, help='simulator server port')
simulator_parser.add_argument('--wrapper', type=str, default='Flask', help='Supported Server: [Flask, FastAPI]')
simulator_parser.add_argument('--url_path', type=str, default='copywriting/urls', help='the path of the saved urls.')
simulator_parser.add_argument('--inference_path', type=str, default='copywriting/data', help='the path of the saved urls.')
simulator_parser.set_defaults(func=simulator)

args = parser.parse_args()
args.func(args)