import time
import tiktoken
from openai import OpenAI
import openai
import os
import anthropic
import json

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00 / 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
    }
    return (
        sum([costmap_in[m] * TOKENS_IN[m] for m in TOKENS_IN])
        + sum([costmap_out[m] * TOKENS_OUT[m] for m in TOKENS_OUT])
    )

# ----------------------------------------------------------------------
# 1) MODEL ALIASES & GUARD CLAUSES
# ----------------------------------------------------------------------

# Model aliases map any variant or synonym to a canonical model name
MODEL_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt4omini": "gpt-4o-mini",
    "gpt-4omini": "gpt-4o-mini",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt4o": "gpt-4o",
    # Note the difference in costmap key vs. the older string:
    "claude-3.5-sonnet": "claude-3-5-sonnet",  
}

# All canonical models recognized in the script
VALID_MODELS = {
    "gpt-4o-mini",
    "gpt-4o",
    "o1-preview",
    "o1-mini",
    "claude-3-5-sonnet",
    "deepseek-chat",
    "o1",
}

# ----------------------------------------------------------------------
# 2) HELPER FUNCTIONS FOR COMMON PATTERNS
# ----------------------------------------------------------------------

def build_messages(system_prompt, prompt, role="user", single_user_msg=False):
    """
    Creates the right shape of messages depending on whether we want
    system+user style or just user style (e.g. 'o1' models).
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
    Wraps logic for openai-based completion calls with or without
    using the more modern `OpenAI()` client or the older `openai.ChatCompletion`.
    """
    if version == "0.28":
        # Legacy approach
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
        # Modern approach with the `OpenAI` client
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
    """
    Extracts answer from an OpenAI completion response.
    """
    return completion.choices[0].message.content

def extract_answer_from_anthropic(message):
    """
    Extract answer from Anthropic-based calls (Claude).
    (Exact parsing depends on how you're retrieving messages)
    """
    # For example:
    return json.loads(message.to_json())["content"][0]["text"]

# ----------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# ----------------------------------------------------------------------

def query_model(model_str, prompt, system_prompt, 
                openai_api_key=None, 
                anthropic_api_key=None, 
                tries=5, 
                timeout=5.0, 
                temp=None, 
                print_cost=True, 
                version="1.5"):

    # 1) Validate & unify model name
    model_str = MODEL_ALIASES.get(model_str, model_str)  # unify known aliases
    if model_str not in VALID_MODELS:
        raise ValueError(f"Unsupported model name or alias: {model_str}")

    # 2) Handle API keys from environment or provided arguments
    preloaded_api = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")

    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # 3) Attempt requests with retries
    for _ in range(tries):
        try:
            # A) GPT-4o-mini
            if model_str == "gpt-4o-mini":
                # Distinguish older version usage or new approach
                # Modern gpt-4o-mini model name might be "gpt-4o-mini-2024-07-18"
                messages = build_messages(system_prompt, prompt)
                if version == "0.28":
                    # older usage
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=temp
                        )
                else:
                    # new usage with the `OpenAI()` client
                    actual_model = "gpt-4o-mini-2024-07-18"
                    completion = openai_create_chat_completion(
                        actual_model, messages, version=version, temp=temp
                    )
                answer = extract_answer_from_openai(completion)

            # B) Claude (Anthropic)
            elif model_str == "claude-3-5-sonnet":
                # Our cost map uses key "claude-3-5-sonnet", unify usage
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                # anthropic v1 calls can differ; adjust as needed
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = extract_answer_from_anthropic(message)

            # C) GPT-4o
            elif model_str == "gpt-4o":
                messages = build_messages(system_prompt, prompt)
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            temperature=temp
                        )
                else:
                    completion = openai_create_chat_completion(
                        "gpt-4o-2024-08-06", messages, version=version, temp=temp
                    )
                answer = extract_answer_from_openai(completion)

            # D) DeepSeek
            elif model_str == "deepseek-chat":
                messages = build_messages(system_prompt, prompt)
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages
                        )
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp
                        )
                answer = extract_answer_from_openai(completion)

            # E) o1-mini
            elif model_str == "o1-mini":
                messages = build_messages(system_prompt, prompt, single_user_msg=True)
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-mini",
                        messages=messages
                    )
                else:
                    completion = openai_create_chat_completion(
                        "o1-mini-2024-09-12", messages, version=version, temp=temp
                    )
                answer = extract_answer_from_openai(completion)

            # F) o1
            elif model_str == "o1":
                messages = build_messages(system_prompt, prompt, single_user_msg=True)
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",
                        messages=messages
                    )
                else:
                    completion = openai_create_chat_completion(
                        "o1-2024-12-17", messages, version=version, temp=temp
                    )
                answer = extract_answer_from_openai(completion)

            # G) o1-preview
            elif model_str == "o1-preview":
                messages = build_messages(system_prompt, prompt, single_user_msg=True)
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-preview",
                        messages=messages
                    )
                else:
                    completion = openai_create_chat_completion(
                        "o1-preview", messages, version=version, temp=temp
                    )
                answer = extract_answer_from_openai(completion)

            # 4) Token counting & cost printing
            try:
                # If an older model name was “absorbed” into a canonical model,
                # you can decide how to handle the encoding here.
                if model_str in ["o1-preview", "o1-mini", "claude-3-5-sonnet", "o1"]:
                    model_for_encoding = "gpt-4o"  # approximate usage
                elif model_str in ["deepseek-chat"]:
                    model_for_encoding = "cl100k_base"
                else:
                    model_for_encoding = model_str

                # Tally the tokens
                chosen_encoding = tiktoken.encoding_for_model(model_for_encoding)
                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0

                TOKENS_IN[model_str] += len(chosen_encoding.encode(system_prompt + prompt))
                TOKENS_OUT[model_str] += len(chosen_encoding.encode(answer))

                if print_cost:
                    print(
                        f"Current experiment cost = ${curr_cost_est()}, "
                        "** Approximate values, may not reflect true cost"
                    )
            except Exception as e:
                if print_cost:
                    print(f"Cost approximation error: {e}")

            return answer

        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue

    raise Exception("Max retries: timeout")


# Example usage:
# print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
