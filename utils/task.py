# utils/task.py
import random
import re
import time
from functools import lru_cache

import yaml
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = config['MODEL']


LOCAL_QWEN_PREFIX = 'Qwen/'
OPENAI_GEMINI_MODEL = 'gemini-3-flash-preview'
OPENAI_GEMINI_BASE_URL = 'https://ai.juguang.chat/v1'
OPENAI_GEMINI_API_KEY = 'sk-dGZZdFacB6dZ79DtPLy4rQd9mq0dxgxUBZ2xa4PIHTsNKZdh'


def _use_openai_gemini_for_qwen():
    import os

    return os.environ.get('TAIRA_USE_OPENAI_GEMINI', '').lower() in {'1', 'true', 'yes', 'on'}


def _chat_with_openai_gemini(messages, temperature=0):
    client = OpenAI(
        api_key=OPENAI_GEMINI_API_KEY,
        base_url=OPENAI_GEMINI_BASE_URL,
    )

    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_GEMINI_MODEL,
                messages=messages,
                max_tokens=32768,
                stream=False,
                temperature=temperature if temperature and temperature > 0 else 1,
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f'[RateLimitError] Retry {attempt + 1}, wait {wait_time:.1f}s...')
            time.sleep(wait_time)
        except (APIError, APIConnectionError) as exc:
            wait_time = base_delay + random.uniform(0, 1)
            print(f'[API/ConnectionError] Retry {attempt + 1}, wait {wait_time:.1f}s...')
            print(exc)
            time.sleep(wait_time)

    raise RuntimeError(f'_chat_with_openai_gemini failed after {max_retries} retries.')


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
    return isinstance(llm_name, str) and llm_name.startswith(LOCAL_QWEN_PREFIX)


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
        if _use_openai_gemini_for_qwen():
            return _chat_with_openai_gemini(messages, temperature=temperature)
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
        if _use_openai_gemini_for_qwen():
            content = _chat_with_openai_gemini(messages, temperature=temperature)
            return content
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
