| 基础模型 | 架构 | 预训练方法 | 中文语料 | 组织机构 |
| --- | --- | --- | --- | --- |
| [**CodeGen-16B**](https://arxiv.org/pdf/2203.13474.pdf) | decoder-only | LM | &#x274C; | **Salesforce** |
| [**BLOOM-7B1**](https://arxiv.org/pdf/2211.05100.pdf) | decoder-only | LM | &#x2705; | **BigScience** |
| [**BLOOMZ-7B1-MT**](https://arxiv.org/pdf/2211.01786.pdf) | decoder-only | LM | &#x2705; | **HuggingFace** |
| [**LLaMA-7B**](https://arxiv.org/pdf/2302.13971.pdf) | decoder-only | LM | &#x2705; | **Salesforce** |
| [**LLaMA-13B**](https://arxiv.org/pdf/2302.13971.pdf) | decoder-only | LM | &#x2705; | **Salesforce** |
| [**GLM-6B**](https://arxiv.org/pdf/2103.10360.pdf) | decoder-only | SpanLM | &#x2705; | **Tsinghua University** |
| [**GLM-130B**](https://arxiv.org/pdf/2210.02414.pdf) | decoder-only | SpanLM | &#x2705; | **Tsinghua University** |
| [**GPT-3**](https://arxiv.org/pdf/2005.14165.pdf) | decoder-only | LM | &#x2705; | **OpenAI** |
| [**GPT-4**](https://arxiv.org/pdf/2303.08774.pdf) | decoder-only | LM | &#x2705; | **OpenAI** |


| 模型名称 | 版本 | 基础模型 | 训练方法 | 中文语料 | 指令调优 | 人类偏好对齐 | 支持多轮 | 组织机构 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [**Aplaca-LoRA-7B**](https://crfm.stanford.edu/2023/03/13/alpaca.html) | `tloen/alpaca-lora-7b` &#x1F917; | **LLaMA-7B** | <font color=green>**LoRA**</font> | &#x274C; | &#x2705; | &#x274C; | &#x274C; | **Stanford University** |
| [**BELLE-7B**](https://arxiv.org/pdf/2303.14742.pdf) | `BelleGroup/BELLE-7B-2M` &#x1F917; | **BLOOMZ-7B1-MT** | <font color=green>**LoRA**</font> | &#x2705; | &#x2705; | &#x274C; | &#x274C; | **Lianjia Tech.** |
| [**ChatGLM-6B**](https://github.com/THUDM/ChatGLM-6B) | `THUDM/chatglm-6b` &#x1F917; | **GLM-6B** | <font color=red>**FT**</font> | &#x2705; | &#x2705; | &#x2705; | &#x2705; | **Tsinghua University** |
| [**Chinese-Alpaca-LoRA-7B**](https://arxiv.org/pdf/2304.08177.pdf) | `ziqingyang/chinese-alpaca-lora-7b` &#x1F917; | **LLaMA-7B** | <font color=green>**LoRA**</font> | &#x2705; | &#x2705; | &#x274C; | &#x274C; | **HIT University** |
| [**Chinese-Vicuna-7B**](https://github.com/Facico/Chinese-Vicuna) | `Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco` &#x1F917; | **LLaMA-7B** | <font color=green>**LoRA**</font> | &#x2705; | &#x2705; | &#x274C; | &#x2705; | **Facico** |
| [**text-davinci-003**](https://arxiv.org/pdf/2005.14165.pdf) | `text-davinci-003` <img src='../assets/icons/openai.svg' style='width:10%'/> | **GPT-3** | <font color=red>**FT**</font> | &#x2705; | &#x2705; | &#x274C; | &#x274C; | **OpenAI** |
| [**gpt-4**](https://arxiv.org/pdf/2303.08774.pdf) | `gpt-4` <img src='../assets/icons/openai.svg' style='width:10%'/> | **GPT-4** | <font color=red>**FT**</font> | &#x2705; | &#x2705; | &#x2705; | &#x2705; | **OpenAI** |
| [**MOSS-moon-003-sft-16B**](https://github.com/OpenLMLab/MOSS) | `fnlp/moss-moon-003-sft` &#x1F917; | **CodeGen-16B** | <font color=red>**FT**</font> | &#x2705; | &#x2705; | &#x2705; | &#x2705; | **Fudan University** |
| [**gpt-3.5-turbo**](https://arxiv.org/pdf/2203.02155.pdf) | `gpt-3.5-turbo` <img src='../assets/icons/openai.svg' style='width:10%'/> | **GPT-3** | <font color=red>**FT**</font> | &#x2705; | &#x2705; | &#x2705; | &#x2705; | **OpenAI** |
| [**Vicuna-7B**](https://lmsys.org/blog/2023-03-30-vicuna/) | `lmsys/vicuna-7b-delta-v1.1` &#x1F917; | **LLaMA-7B** | <font color=green>**LoRA**</font> | &#x274C; | &#x2705; | &#x2705; | &#x2705; | **UC Berkeley+** |
| [**Vicuna-13B**](https://lmsys.org/blog/2023-03-30-vicuna/) | `lmsys/vicuna-13b-delta-v1.1` &#x1F917; | **LLaMA-13B** | <font color=green>**LoRA**</font> | &#x274C; | &#x2705; | &#x2705; | &#x2705; | **UC Berkeley+** |