import os
import time
import json

import openai
import anthropic
from openai import OpenAI
import tiktoken

MODEL_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt4omini": "gpt-4o-mini",
    "gpt-4omini": "gpt-4o-mini",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt4o": "gpt-4o",
    "claude-3.5-sonnet": "claude-3-5-sonnet",
}
VALID_MODELS = {
    "gpt-4o-mini",
    "gpt-4o",
    "o1-preview",
    "o1-mini",
    "claude-3-5-sonnet",
    "deepseek-chat",
    "o1",
}

TOKENS_IN = dict()
TOKENS_OUT = dict()

COSTS_IN = {
    "gpt-4o": 2.50e-6,
    "gpt-4o-mini": 0.150e-6,
    "o1-preview": 15.00e-6,
    "o1-mini": 3.00e-6,
    "claude-3-5-sonnet": 3.00e-6,
    "deepseek-chat": 1.00e-6,
    "o1": 15.00e-6,
}
COSTS_OUT = {
    "gpt-4o": 10.00e-6,
    "gpt-4o-mini": 0.6e-6,
    "o1-preview": 60.00e-6,
    "o1-mini": 12.00e-6,
    "claude-3-5-sonnet": 12.00e-6,
    "deepseek-chat": 5.00e-6,
    "o1": 60.00e-6,
}

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    """
    Summation of input token cost + output token cost so far.
    """
    return (
        sum([COSTS_IN[m] * TOKENS_IN[m] for m in TOKENS_IN if m in COSTS_IN]) +
        sum([COSTS_OUT[m] * TOKENS_OUT[m] for m in TOKENS_OUT if m in COSTS_OUT])
    )

def build_messages(system_prompt, prompt, role="user", single_user_msg=False):
    """
    Build the messages list.
    If single_user_msg is True, concatenate system_prompt and prompt into one message.
    Otherwise, return a two-message list with system and user roles.
    """
    if single_user_msg:
        return [{"role": "user", "content": system_prompt + prompt}]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": role, "content": prompt},
        ]

def openai_create_chat_completion(model, messages, version="1.5", temp=None):
    """
    Wrapper for creating chat completions using either the legacy or new OpenAI API.
    """
    if version == "0.28":
        if temp is None:
            return openai.ChatCompletion.create(
                model=model,
                messages=messages
            )
        else:
            return openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temp
            )
    else:
        client = OpenAI()
        if temp is None:
            return client.chat.completions.create(
                model=model,
                messages=messages
            )
        else:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp
            )

def extract_answer_from_openai(completion):
    return completion.choices[0].message.content

def extract_answer_from_anthropic(message):
    return json.loads(message.to_json())["content"][0]["text"]

def _update_token_usage_and_print_cost(model_name, system_prompt, user_prompt, answer, print_cost=True):
    """
    Helper to update global token usage counters for the given model,
    and optionally print the approximate cost so far.
    """
    try:
        if model_name in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
            encoding_to_use = tiktoken.encoding_for_model("gpt-4o")
        elif model_name in ["deepseek-chat"]:
            encoding_to_use = tiktoken.encoding_for_model("cl100k_base")
        else:
            encoding_to_use = tiktoken.encoding_for_model(model_name)

        if model_name not in TOKENS_IN:
            TOKENS_IN[model_name] = 0
            TOKENS_OUT[model_name] = 0

        TOKENS_IN[model_name] += len(encoding_to_use.encode(system_prompt + user_prompt))
        TOKENS_OUT[model_name] += len(encoding_to_use.encode(answer))

        if print_cost:
            print(f"Current experiment cost = ${curr_cost_est():.4f}, ** Approximate values, may not reflect true cost")

    except Exception as e:
        if print_cost:
            print(f"Cost approximation has an error? {e}")

def query_model(model_str, prompt, system_prompt, openai_api_key=None, anthropic_api_key=None,
                tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    model_str = MODEL_ALIASES.get(model_str, model_str)
    if model_str not in VALID_MODELS:
        raise ValueError(f"Unsupported model name or alias: {model_str}")

    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini":
                messages = build_messages(system_prompt, prompt)
                if version == "0.28":
                    completion = openai_create_chat_completion(model_str, messages, version=version, temp=temp)
                else:
                    completion = openai_create_chat_completion("gpt-4o-mini-2024-07-18", messages, version=version, temp=temp)
                answer = extract_answer_from_openai(completion)

            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = extract_answer_from_anthropic(message)

            elif model_str == "gpt-4o":
                messages = build_messages(system_prompt, prompt)
                if version == "0.28":
                    completion = openai_create_chat_completion(model_str, messages, version=version, temp=temp)
                else:
                    completion = openai_create_chat_completion("gpt-4o-2024-08-06", messages, version=version, temp=temp)
                answer = extract_answer_from_openai(completion)

            elif model_str == "deepseek-chat":
                messages = build_messages(system_prompt, prompt)
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = extract_answer_from_openai(completion)

            elif model_str == "o1-mini":
                messages = build_messages(system_prompt, prompt, single_user_msg=True)
                if version == "0.28":
                    completion = openai_create_chat_completion(model_str, messages, version=version, temp=temp)
                else:
                    completion = openai_create_chat_completion("o1-mini-2024-09-12", messages, version=version, temp=temp)
                answer = extract_answer_from_openai(completion)

            elif model_str == "o1":
                messages = build_messages(system_prompt, prompt, single_user_msg=True)
                if version == "0.28":
                    completion = openai_create_chat_completion("o1-2024-12-17", messages, version=version, temp=temp)
                else:
                    completion = openai_create_chat_completion("o1-2024-12-17", messages, version=version, temp=temp)
                answer = extract_answer_from_openai(completion)

            elif model_str == "o1-preview":
                messages = build_messages(system_prompt, prompt, single_user_msg=True)
                if version == "0.28":
                    completion = openai_create_chat_completion(model_str, messages, version=version, temp=temp)
                else:
                    completion = openai_create_chat_completion("o1-preview", messages, version=version, temp=temp)
                answer = extract_answer_from_openai(completion)

            # Now update token usage and optionally print cost:
            _update_token_usage_and_print_cost(
                model_name=model_str,
                system_prompt=system_prompt,
                user_prompt=prompt,
                answer=answer,
                print_cost=print_cost
            )

            return answer

        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue

    raise Exception("Max retries: timeout")

# print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
