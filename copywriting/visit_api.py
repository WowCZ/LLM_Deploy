import requests
import json
from multiprocessing import Pool

def post_data(url_ip, header, port, data):
    url = f'http://{url_ip}:{port}/generate'
    print(url)
    reqs = requests.post(url=url, headers=header, data=json.dumps(data))

    return json.loads(reqs.text)

def _post_data(args):
    url_ip, header, port, data = args
    return post_data(url_ip, header, port, data)

def multiprocess_post(url_ip, header, ports, datas):
    pool_size = len(ports)
    with Pool(pool_size) as p:
        if type(url_ip) is str:
            url_ip = [url_ip]*pool_size
        mp_reqs = p.map(_post_data, zip(url_ip, [header]*pool_size, ports, datas))

    return mp_reqs

def visit_llm(url_ip, header, port, data):
    if type(port) is str:
        return post_data(url_ip, header, port, data)
    else:
        return multiprocess_post(url_ip, header, port, data)


if __name__ == '__main__':
    # Single Thread
    url_ip = '43.130.133.215'
    port = '6501'
    header = {'Content-Type': 'application/json'}
    data = {
        "prompt": "有这样一个故事：“我：“爸在干嘛呢？最近家里生意还好吧。”爸：“已汇””，请问这个故事的笑点在哪儿？"
    }

    print(post_data(url_ip, header, port, data))

    # # Multiprocessing
    # ports = ['5001', '5002']
    # datas = [
    #     {
    #     "prompt": "'有这样一个故事：“我：“爸在干嘛呢？最近家里生意还好吧。”爸：“已汇””，请问这个故事的笑点在哪儿？'"
    #     },
    #     {
    #     "prompt": "有这样一个故事，““爸，端午节我不回家，捎几个粽子给我吧。”“行，你要哪种？”“都行，能解馋就行。”“好！”晚上回宿舍，打开邮箱发现爸爸发了一封邮件，足足4个G，下面还留言：不知道你好哪口，就每种给你发了一个。”，请问这个故事的笑点在哪儿？"
    #     }
    # ]
    # print(multiprocess_post(url_ip, header, ports, datas))