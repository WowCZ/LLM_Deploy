# -*- coding:utf-8 -*-

# @Author:      zp
# @Time:        2023/4/7 15:19

import openai
import pandas as pd

from enum import Enum
from openai import util
from base64 import b64encode

# 这里是使用gpt4 进行的请求
openai.api_base = "https://ai-proxy.shlab.tech/v4"
username = b"SMUDSOLHLZ"
password = b"##=%%u-%#kj+huxpbrh^"
model="gpt-4"


class ApiType(Enum):
    AZURE = 1
    OPEN_AI = 2
    AZURE_AD = 3

    @staticmethod
    def from_str(label):
        if label.lower() == "azure":
            return ApiType.AZURE
        elif label.lower() in ("azure_ad", "azuread"):
            return ApiType.AZURE_AD
        elif label.lower() in ("open_ai", "openai"):
            return ApiType.OPEN_AI
        else:
            raise openai.error.InvalidAPIType(
                "The API type provided in invalid. Please select one of the supported API types: 'azure', 'azure_ad', 'open_ai'"
            )

util.api_key_to_header = (
    lambda api, key: {"Authorization": f"Bearer {key}"}
    if api in (ApiType.OPEN_AI, ApiType.AZURE_AD)
    else {"Authorization": f"{key}"}
)

def get_token(username, password):
    authstr = "Basic " + to_native_string(
            b64encode(b":".join((username, password))).strip()
        )
    return authstr

def to_native_string(string, encoding="ascii"):
    """Given a string object, regardless of type, returns a representation of
    that string in the native string type, encoding and decoding where
    necessary. This assumes ASCII unless told otherwise.
    """
    if isinstance(string, str):
        out = string
    else:
        out = string.decode(encoding)

    return out

def chat(message: str, model="gpt-3.5-turbo") -> str:
    """
    model: gpt-3.5-turbo, gpt-3.5-turbo-0301
    """
    message_log = [{"role": "user", "content": message}]
    completion = openai.ChatCompletion.create(model=model, messages=message_log)
    res = completion.choices[0].message.content
    print(f"question: {message} ===> answer: {res}")
    return res


def get_input(file_name="./input.xlsx") -> list:
    df = pd.read_excel(file_name, sheet_name='Sheet1')
    return df["question"].tolist()


def save_output(data: dict, file_name="./output.xlsx"):
    df = pd.DataFrame(data)
    # 将 DataFrame 对象保存到 Excel 文件
    df.to_excel(file_name, index=False)


if __name__ == "__main__":
    openai.api_key = get_token(username, password)
    result = []
    datas = ['中国有多少个省份？']
    for one_qa in datas:
        ress = chat(one_qa, model=model)
        result.append({"question": one_qa, "answer": ress})
    print(result)