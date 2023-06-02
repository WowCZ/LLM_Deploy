import json
from typing import Union

def record_as_json(json_file: str, data:Union[dict, list]):
    ori_data = json.load(open(json_file))
    if len(ori_data) == 0:
        ori_data = data
    else:
        if type(data) is dict:
            for k, v in data.items():
                ori_data[k] = v

        elif type(data) is list:
            ori_data.extend(data)

        else:
            raise TypeError(f'{type(data)} is not given.')
