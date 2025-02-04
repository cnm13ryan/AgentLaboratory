import os
import sys
import random
import logging
import warnings

from abc import abstractmethod
from contextlib import contextmanager

# Local/Project Imports
from copy import copy
from tools import *
from inference import *

@contextmanager
def suppress_stdout():
    """
    Temporarily redirect stdout to devnull.
    Useful to suppress unwanted output during code execution.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


os.environ["JOBLIB_VERBOSITY"] = "0"
logging.basicConfig(level=logging.WARNING)
logging.getLogger('sklearn.model_selection').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


GLOBAL_REPAIR_ATTEMPTS = 2


class Command:
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        """
        Returns a description of the command.
        """
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        """
        Executes the command with given arguments.
        """
        pass

    @abstractmethod
    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the command string matches this command.
        """
        pass

    @abstractmethod
    def parse_command(self, cmd_str) -> tuple:
        """
        Parses the command string into its components.
        """
        pass


"""
@@@@@@@@@@@@@@@@@@
@@ CODING TOOLS @@
@@@@@@@@@@@@@@@@@@
"""

class Replace(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-replace"

    def docstring(self) -> str:
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "You also have access to a code replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current code and erase all existing code.\n"
            "You can use this tool via the following command: ```REPLACE\n<code here>\n```, where REPLACE is the word REPLACE and <code here> will be the new code that is replacing the entire set of old code. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Try limiting the use of rewriting and aim for editing the code more."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> new code
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        if "```REPLACE" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"
        code_ret = execute_code(code_exec)
        if "[CODE EXECUTION ERROR]" in code_ret: return False, (None, code_ret,)
        return True, (new_code.split("\n"), code_ret)



class Edit(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-edit"

    def docstring(self) -> str:
        return (
            "============= CODE EDITING TOOL =============\n"
            "You also have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Your changes should significantly change the functionality of the code."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> N (int)
        # args[1] -> M (int)
        # args[2] -> old code
        # args[3] -> new lines to replace
        # args[4] -> new lines to replace
        try:
            args = args[0]
            current_code = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1]+1)))
            for _ln in lines_to_replace:
                current_code.pop(_ln)
            for _line in lines_to_add:
                current_code.insert(args[0], _line)
            new_code = "\n".join(current_code)
            code_exec = f"{args[4]}\n{new_code}"
            code_ret = execute_code(code_exec)
            if "CODE EXECUTION ERROR" in code_ret: return (False, None, code_ret)
            return (True, current_code, code_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        if "```EDIT" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        cmd_str, codelines, datasetcode = args[0], args[1], args[2]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0: return False, None
            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2: return False, None
            lines_to_edit = [int(_) for _ in lines_to_edit]
            if len(text[1:]) == 0: return False, None
            return success, (lines_to_edit[0], lines_to_edit[1], codelines, text[1:], datasetcode)
        except Exception as e:
            return False, (None, None, None, None, None)


def get_score(outlined_plan, code, code_return, REWARD_MODEL_LLM, attempts=3, openai_api_key=None):
    e = str()
    for _attempt in range(attempts):
        try:
            # todo: have a reward function here
            sys = (
                f"You are a professor agent who is serving as an expert reward model that can read a research plan, research code, and code output and are able to determine how well a model followed the plan, built the code, and got the proper output scored from 0 to 1 as a float.\n\n"
                f"You must structure your score exactly in the following way: ```SCORE\n<score here>\n``` where SCORE is just the word score, <score here> is a floating point number between 0 and 1 representing how well the model followed the plan, built the code, and got the proper output."
            )
            scoring = query_model(
                model_str=f"{REWARD_MODEL_LLM}",
                system_prompt=sys,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Outlined in the following text is the research plan that the machine learning engineer was tasked with building: {outlined_plan}\n\n"
                    f"The following text is the research code that the model produced: \n{code}\n\n"
                    f"The following is the output from the model: {code_return}\n\n"), temp=0.6)
            performance = extract_prompt(text=scoring, word="SCORE")
            performance = float(performance)
            return performance, f"The performance of your submission is: {performance}", True
        except Exception as e:
            return None, str(e), False
    return 0, e


def code_repair(code, error, ctype, REPAIR_LLM, openai_api_key=None):
    if ctype == "replace":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
            "Your output should match the original code as closely as possible.\n"
            "You must wrap the code in the following ```python\n<code here>\n```\n"
            "Do not forget the opening ```python and the closing ```."
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}", temp=0.8)
        return extract_prompt(model_resp, "python")
    elif ctype == "edit":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
            "Your output should match the original code as closely as possible.\n"
            
            "============= CODE EDITING TOOL =============\n"
            "You have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code.\n"
            "Please use the code editing tool to fix this code."
            "Do not forget the opening ```EDIT N M and the closing ```."
            "Your output should look like the following\n\n```EDIT N M\n<new lines to replace old lines>\n```"
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}", temp=0.2)
        return model_resp


class MLESolver:
    """
    Orchestrates the code generation and editing process to solve a machine learning research plan.
    """
    def __init__(self, dataset_code, openai_api_key=None, notes=None, max_steps=10, insights=None, plan=None, llm_str=None):
        self.notes = notes if notes is not None else []
        self.dataset_code = dataset_code
        self.plan = plan if plan is not None else ""
        self.llm_str = llm_str
        self.verbose = False
        self.max_codes = 2
        self.st_hist_len = 2
        self.min_gen_trials = 2
        self.code_lines = str()
        self.st_history = []
        self.insights = insights
        self.code_reflect = ""
        self.max_steps = max_steps
        self.prev_code_ret = ""
        self.should_execute_code = True
        self.openai_api_key = openai_api_key
        self.best_score = None
        self.best_codes = []
        self.prev_working_code = ""

    # --- Initialization Methods ---
    def initial_solve(self):
        """
        Initialize the solver by generating the initial code using the REPLACE command.
        """
        self.best_score = None
        self.commands = [Replace()]
        self.model = self.llm_str

        init_code, init_return, self.best_score = self.gen_initial_code()
        self.best_codes = [(copy(init_code), self.best_score, init_return)]

        self.code_lines = init_code
        self.model = self.llm_str
        self.commands = [Edit(), Replace()]
        self.prev_working_code = copy(self.code_lines)

    def gen_initial_code(self):
        """
        Generate the initial code by querying the model until a valid solution is found.
        """
        num_attempts = 0
        error_hist = []

        while True:
            if num_attempts == 0:
                err_hist = ""
            else:
                err = f"Previous attempt failed with errors:\n{error_hist}\nAvoid repeating them."
                err_hist = err

            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"{err_hist}\nUse ```REPLACE to create the initial code.\nPlease enter the ```REPLACE command below:\n",
                temp=1.0
            )
            model_resp = self.clean_text(model_resp)
            cmd_str, code_lines, prev_code_ret, exec_code, score = self.process_command(model_resp)

            print(f"@@@ INIT ATTEMPT: {num_attempts} --> {cmd_str}")
            print(f"$$$ Score: {score}")

            if score is not None:
                break

            error_hist.append(cmd_str)
            if len(error_hist) > 5:
                error_hist.pop(0)

            num_attempts += 1

        return code_lines, prev_code_ret, score

    # --- Main Solving Loop ---
    def solve(self):
        """
        The main loop that queries for commands and selects the best code solution.
        """
        num_attempts = 0
        best_pkg = None
        top_score = None
        self.prev_code_ret = None
        self.should_execute_code = False

        while True:
            cmd_app_str = "You must output either the ```EDIT or ```REPLACE command immediately. " if len(self.commands) == 2 else ""
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"The following is your history:\n{self.history_str()}\n\n{cmd_app_str}Now please enter a command:\n",
                temp=1.0
            )
            model_resp = self.clean_text(model_resp)
            self.code_lines = copy(random.choice(self.best_codes)[0])
            cmd_str, code_lines, prev_code_ret, exec_code, score = self.process_command(model_resp)
            self.st_history.append([model_resp, prev_code_ret, code_lines, cmd_str])
            if len(self.st_history) > self.st_hist_len:
                self.st_history.pop(0)

            if score is not None:
                if top_score is None or score > top_score:
                    best_pkg = (copy(code_lines), copy(prev_code_ret), copy(exec_code), copy(model_resp), copy(cmd_str))
                    top_score = score

            print(f"@@@ Command Exec // Attempt {num_attempts}: {cmd_str}")
            print(f"$$$ Score: {score}")

            if num_attempts >= self.min_gen_trials and top_score is not None:
                break
            num_attempts += 1

        self.code_lines, self.prev_code_ret, self.should_execute_code, model_resp, cmd_str = best_pkg

        if top_score is not None and (not self.best_codes or top_score > self.best_codes[-1][1]):
            if len(self.best_codes) >= self.max_codes:
                self.best_codes.pop(-1)
                self.code_reflect = self.reflect_code()
            self.best_codes.append((copy(self.code_lines), copy(top_score), self.prev_code_ret))
            self.best_codes.sort(key=lambda x: x[1], reverse=True)

        return model_resp, cmd_str

    # --- Reflection and Feedback ---
    def reflect_code(self):
        """
        Use the language model to reflect on past solutions and suggest improvements.
        """
        sep_line = f"{'$' * 40}\n\n"
        code_strs = ""
        for c_lines, sc, ret in self.best_codes:
            snippet = self.generate_code_lines(c_lines)
            code_strs += f"{snippet}\nCode Return {sc}\n{sep_line}"
        prompt_text = "Please reflect on the following code sets and propose improvements."
        system_msg = self.system_prompt(commands=False) + code_strs

        return query_model(
            prompt=prompt_text,
            system_prompt=system_msg,
            model_str=self.llm_str,
            openai_api_key=self.openai_api_key
        )

    def process_command(self, model_resp):
        """
        Process and execute the command (EDIT or REPLACE) from the model response.
        Returns a tuple:
            (cmd_str, code_lines, prev_code_ret, should_execute_code, score)
        """
        prev_code_ret = self.prev_code_ret
        should_execute_code = self.should_execute_code
        code_lines = copy(self.code_lines)
        remove_figures()

        with suppress_stdout():
            for cmd in self.commands:
                if cmd.matches_command(model_resp):
                    # [Process CODE-edit and CODE-replace similarly...]
                    # (Details omitted here for brevity; see full version for complete logic)
                    pass
            print("$$$$ INVALID COMMAND (failed)")
            return "Command not supported, choose from existing commands", None, None, None, None

    def feedback(self, code_return):
        """
        Provide execution feedback and generate a reflection.
        """
        # [Feedback logic here...]
        pass

    def reflection(self, reflect_prompt, code_return):
        """
        Generate a reflection based on the provided prompt.
        """
        code_str = self.generate_code_lines(self.code_lines)
        system_msg = self.system_prompt(commands=False)
        refl = query_model(
            prompt=reflect_prompt,
            system_prompt=system_msg,
            model_str=self.llm_str,
            openai_api_key=self.openai_api_key
        )
        return (
            f"During the previous execution, the code was:\n{code_str}\n\n"
            f"Code returned:\n{code_return}\n\n"
            f"Reflection:\n{refl}\n"
        )

    # --- Utility Methods ---
    def history_str(self):
        """
        Return a formatted history string of previous attempts.
        """
        hist_str = ""
        for idx, item in enumerate(self.st_history):
            model_resp, code_ret, lines, cmd_str = item
            hist_str += f"-------- History ({len(self.st_history)-idx} steps ago) -----\n"
            hist_str += f"LM Response:\n{model_resp}\n"
            hist_str += f"COMMAND Output:\n{cmd_str}\n"
            hist_str += f"Code:\n{'#'*20}\n{lines}\n{'#'*20}\n\n"
            hist_str += f"Feedback/Reflection:\n{code_ret}\n"
            hist_str += f"-------- End of history ({len(self.st_history)-idx} steps ago) -------\n\n"
        return hist_str

    @staticmethod
    def generate_code_lines(code):
        """
        Generate code with line numbers.
        """
        return "\n".join(f"{i} |{line}" for i, line in enumerate(code))

    def system_prompt(self, commands=True):
        """
        Build the system prompt for the language model.
        """
        result = (
            f"{self.role_description()}\n"
            f"The following are your task instructions: {self.phase_prompt()}\n"
            f"Literature review insights:\n{self.insights}\n"
            f"{self.code_reflect}\n"
            f"Notes:\n{self.notes}\n"
            f"Plan:\n{self.plan}\n"
            f"{self.generate_dataset_descr_prompt()}\n"
            "Generate at least two figures (Figure_1.png, Figure_2.png).\n"
            "Ensure accuracy is not 0%.\n"
            "Include detailed print statements to explain results.\n"
        )
        if commands:
            result += (
                f"The following are the commands available: {self.command_descriptions()}\n"
                "Only execute a single command at a time.\n"
            )
        return result

    def generate_dataset_descr_prompt(self):
        """
        Return the dataset description prompt.
        """
        return f"The following dataset code is automatically prepended:\n{self.dataset_code}"

    @staticmethod
    def phase_prompt():
        return (
            "You are an ML engineer writing code for a research project.\n"
            "Your code should execute all experiments in one file and use pre-installed libraries only.\n"
            "Focus on clarity and correctness."
        )

    @staticmethod
    def role_description():
        return (
            "You are an expert machine learning engineer at a top university, solving complex ML challenges."
        )

    @staticmethod
    def _common_code_errors():
        return (
            "Import everything you use.\n"
            "Avoid bugs by reflecting on your code before execution.\n"
            "Use commands exactly as specified (EDIT or REPLACE), and never execute multiple commands at once.\n"
            "Avoid TensorFlow/Keras; use PyTorch or scikit-learn."
        )

    def command_descriptions(self):
        cmd_strings = "\n".join([c.docstring() for c in self.commands])
        instructions = (
            "Commands must be in the form:\n"
            "```COMMAND\n<content>\n```\n"
            "where COMMAND is EDIT or REPLACE. Execute only one command at a time.\n"
        )
        return instructions + self._common_code_errors() + "\n" + cmd_strings

    def run_code(self):
        """
        Execute the generated code if changes have been made.
        """
        if self.prev_code_ret is not None:
            return self.prev_code_ret
        elif self.should_execute_code:
            return execute_code("\n".join(self.code_lines))
        return "No changes have been made to the code."
