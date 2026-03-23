# utils/task.py
import re
from functools import lru_cache

import yaml
from openai import OpenAI

with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = config['MODEL']


def _load_runtime_config():
    with open('system_config.yaml') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


@lru_cache(maxsize=2)
def _load_local_qwen(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map='auto',
    )
    return tokenizer, model


def _is_qwen_local_model(llm_name):
    return isinstance(llm_name, str) and llm_name.startswith('Qwen/')


def _chat_with_local_qwen(messages, llm_name, temperature=0):
    runtime_config = _load_runtime_config()
    tokenizer, local_model = _load_local_qwen(llm_name)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=runtime_config.get('QWEN_ENABLE_THINKING', True),
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(local_model.device)

    generation_kwargs = {
        'max_new_tokens': runtime_config.get('QWEN_MAX_NEW_TOKENS', 2048),
        'pad_token_id': tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        generation_kwargs['temperature'] = temperature
        generation_kwargs['do_sample'] = True
    else:
        generation_kwargs['do_sample'] = False

    generated_ids = local_model.generate(**model_inputs, **generation_kwargs)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip('\n')
    return content



def get_completion(messages, llm=model, temperature=0):  # claude-3-5-sonnet-20240620 gpt-4o-2024-08-06 qwen-plus
    runtime_config = _load_runtime_config()
    llm_name = llm or runtime_config['MODEL']

    if _is_qwen_local_model(llm_name):
        return _chat_with_local_qwen(messages, llm_name, temperature=temperature)

    client = OpenAI(
        base_url=runtime_config.get('OPENAI_BASE_URL') or None,
        api_key=runtime_config.get('OPENAI_API_KEY') or None,
    )
    tokens = 5000
    response = client.chat.completions.create(
        model=llm_name,
        messages=messages,
        temperature=temperature if temperature else 1,
        timeout=50,
        max_tokens=tokens,
        top_p=0.1,
    )

    return response.choices[0].message.content



def get_json(messages, json_format, llm=model, temperature=0):
    runtime_config = _load_runtime_config()
    llm_name = llm or runtime_config['MODEL']

    if _is_qwen_local_model(llm_name):
        content = _chat_with_local_qwen(messages, llm_name, temperature=temperature)
        return content

    client = OpenAI(
        base_url=runtime_config.get('OPENAI_BASE_URL') or None,
        api_key=runtime_config.get('OPENAI_API_KEY') or None,
    )

    response = client.beta.chat.completions.parse(
        model=llm_name,
        messages=messages,
        temperature=temperature,
        timeout=50,
        max_tokens=1000,
        response_format=json_format,
    )

    return response.choices[0].message.content



def extract_braces_content(s):
    s = s.replace("\\'", "'")
    # 使用正则表达式匹配最前面的{和最后面的}之间的所有内容
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None
