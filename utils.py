import os
import re
import shutil
import subprocess

import tiktoken


# Constant holding extra LaTeX packages for insertion
LATEX_PACKAGES = (
    "\n\\usepackage{amsmath}"
    "\n\\usepackage{amssymb}"
    "\n\\usepackage{array}"
    "\n\\usepackage{algorithm}"
    "\n\\usepackage{algorithmicx}"
    "\n\\usepackage{algpseudocode}"
    "\n\\usepackage{booktabs}"
    "\n\\usepackage{colortbl}"
    "\n\\usepackage{color}"
    "\n\\usepackage{enumitem}"
    "\n\\usepackage{fontawesome5}"
    "\n\\usepackage{float}"
    "\n\\usepackage{graphicx}"
    "\n\\usepackage{hyperref}"
    "\n\\usepackage{listings}"
    "\n\\usepackage{makecell}"
    "\n\\usepackage{multicol}"
    "\n\\usepackage{multirow}"
    "\n\\usepackage{pgffor}"
    "\n\\usepackage{pifont}"
    "\n\\usepackage{soul}"
    "\n\\usepackage{sidecap}"
    "\n\\usepackage{subcaption}"
    "\n\\usepackage{titletoc}"
    "\n\\usepackage[symbol]{footmisc}"
    "\n\\usepackage{url}"
    "\n\\usepackage{wrapfig}"
    "\n\\usepackage{xcolor}"
    "\n\\usepackage{xspace}"
)


def compile_latex(latex_code, should_compile=True, output_filename="output.pdf", timeout=30):
    """
    Compile LaTeX code into a PDF using pdflatex.

    :param latex_code: The raw LaTeX source code as a string
    :param should_compile: Whether or not to run the pdflatex compilation
    :param output_filename: Name of the output PDF file
    :param timeout: Maximum time (in seconds) for the compilation process
    :return: A string describing success or any encountered error
    """
    # Insert additional packages right after \documentclass{article}
    latex_code = latex_code.replace(r"\documentclass{article}", r"\documentclass{article}" + LATEX_PACKAGES)

    tex_directory = "research_dir/tex"
    temp_tex_path = os.path.join(tex_directory, "temp.tex")

    # Write the LaTeX code to temp.tex in the specified directory
    with open(temp_tex_path, "w") as f:
        f.write(latex_code)

    if not should_compile:
        return "Compilation successful (skipped actual compile)"

    try:
        # Compiling the LaTeX code using pdflatex
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
            check=True,                    # Raises CalledProcessError on non-zero exit codes
            stdout=subprocess.PIPE,        # Capture stdout
            stderr=subprocess.PIPE,        # Capture stderr
            timeout=timeout,               # Timeout for the process
            cwd=tex_directory
        )
        return f"Compilation successful: {result.stdout.decode('utf-8')}"

    except subprocess.TimeoutExpired:
        return f"[CODE EXECUTION ERROR]: Compilation timed out after {timeout} seconds"

    except subprocess.CalledProcessError as e:
        return (
            f"[CODE EXECUTION ERROR]: Compilation failed: "
            f"{e.stderr.decode('utf-8')} {e.output.decode('utf-8')}. "
            "There was an error in your LaTeX."
        )


def count_tokens(messages, model="gpt-4"):
    """
    Count the total number of tokens across a list of message dictionaries.
    """
    encoder = tiktoken.encoding_for_model(model)
    return sum(len(encoder.encode(msg["content"])) for msg in messages)


def remove_figures():
    """
    Remove all PNG files in the current directory whose names start with 'Figure_'.
    """
    for filename in os.listdir("."):
        if filename.startswith("Figure_") and filename.endswith(".png"):
            os.remove(filename)


def remove_directory(dir_path):
    """
    Remove a directory if it exists; otherwise, print that it doesn't exist.
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' removed successfully.")
        except Exception as e:
            print(f"Error removing directory '{dir_path}': {e}")
    else:
        print(f"Directory '{dir_path}' does not exist or is not a directory.")


def save_to_file(location, filename, data):
    """
    Utility function to save text data to a specified file path.
    """
    filepath = os.path.join(location, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(data)
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving file '{filename}': {e}")


def clip_tokens(messages, model="gpt-4", max_tokens=100000):
    """
    Clip tokens from the start of a message list if exceeding max_tokens,
    preserving message boundaries as much as possible.
    """
    encoder = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoder.encode(msg["content"])) for msg in messages)

    if total_tokens <= max_tokens:
        return messages  # No need to clip if under the limit

    # Convert all messages to a list of tokens
    tokenized_messages = []
    for msg in messages:
        tokenized_content = encoder.encode(msg["content"])
        tokenized_messages.append({"role": msg["role"], "content": tokenized_content})

    # Flatten all tokens and remove from the beginning
    all_tokens = [tok for m in tokenized_messages for tok in m["content"]]
    clipped_tokens = all_tokens[total_tokens - max_tokens:]

    # Rebuild the clipped messages
    clipped_messages = []
    current_idx = 0
    for msg in tokenized_messages:
        count = len(msg["content"])
        if current_idx + count > len(clipped_tokens):
            clipped_content = clipped_tokens[current_idx:]
            clipped_messages.append({
                "role": msg["role"],
                "content": encoder.decode(clipped_content)
            })
            break
        else:
            clipped_content = clipped_tokens[current_idx:current_idx + count]
            clipped_messages.append({
                "role": msg["role"],
                "content": encoder.decode(clipped_content)
            })
            current_idx += count

    return clipped_messages


def extract_prompt(text, word):
    """
    Extract text enclosed between triple backticks of a specific language/format.
    """
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    return "\n".join(code_blocks).strip()
