import os
import sys
import random
import logging
import warnings

from abc import abstractmethod
from contextlib import contextmanager

# Local/Project Imports
from copy import copy
from tools import *          # Used for extract_prompt, execute_code, remove_figures, etc.
from inference import *      # Used for query_model
# ---------------------------------------------------

# ------------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------------
os.environ["JOBLIB_VERBOSITY"] = "0"
logging.basicConfig(level=logging.WARNING)
logging.getLogger('sklearn.model_selection').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

GLOBAL_REPAIR_ATTEMPTS = 2


# ------------------------------------------------------------------
# OUTPUT SUPPRESSION CONTEXT MANAGER
# ------------------------------------------------------------------
@contextmanager
def suppress_stdout():
    """
    Temporarily redirect stdout to devnull.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ------------------------------------------------------------------
# BASE COMMAND CLASS
# ------------------------------------------------------------------
class Command:
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        pass

    @abstractmethod
    def matches_command(self, cmd_str) -> bool:
        pass

    @abstractmethod
    def parse_command(self, cmd_str) -> tuple:
        pass


# ------------------------------------------------------------------
# REPLACE COMMAND
# ------------------------------------------------------------------
class Replace(Command):
    """
    Allows rewriting/replacing the entire code (in one block).
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-replace"

    def docstring(self) -> str:
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "You can rewrite/replace all of the current code, erasing the existing code.\n"
            "Usage: ```REPLACE\n<code here>\n```\n"
            "This is useful for making very significant changes.\n"
            "Before finalizing, the new code is tested; if it errors, it will NOT replace the old code."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> new code
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        return "```REPLACE" in cmd_str

    def parse_command(self, *args) -> tuple:
        """
        Extracts the new code from the command, executes it, and returns
        whether it succeeded plus any outputs.
        """
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"
        code_ret = execute_code(code_exec)
        if "[CODE EXECUTION ERROR]" in code_ret:
            return False, (None, code_ret,)
        return True, (new_code.split("\n"), code_ret)


# ------------------------------------------------------------------
# EDIT COMMAND
# ------------------------------------------------------------------
class Edit(Command):
    """
    Allows replacing lines n through m (inclusive) with any number of new lines.
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-edit"

    def docstring(self) -> str:
        return (
            "============= CODE EDITING TOOL =============\n"
            "Usage: ```EDIT N M\n<new lines>\n``` replaces lines N through M.\n"
            "Before finalizing, the new code is tested; if it errors, it will NOT replace the old code."
        )

    def execute_command(self, *args) -> str:
        """
        args[0] = N
        args[1] = M
        args[2] = old code
        args[3] = new lines to insert
        args[4] = dataset code (for execution)
        """
        try:
            args = args[0]
            n_line, m_line = args[0], args[1]
            current_code = args[2]
            new_lines = list(reversed(args[3]))
            dataset_code = args[4]

            # Remove existing lines in [n_line, m_line]
            for line_idx in reversed(range(n_line, m_line + 1)):
                current_code.pop(line_idx)

            # Insert new lines at position n_line
            for line in new_lines:
                current_code.insert(n_line, line)

            updated_code = "\n".join(current_code)
            code_exec = f"{dataset_code}\n{updated_code}"
            code_ret = execute_code(code_exec)

            if "CODE EXECUTION ERROR" in code_ret:
                return (False, None, code_ret)

            return (True, current_code, code_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        return "```EDIT" in cmd_str

    def parse_command(self, *args) -> tuple:
        """
        Extract the line range and new lines from the command text.
        """
        cmd_str, codelines, datasetcode = args[0], args[1], args[2]
        success = True
        try:
            # "EDIT" block
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if not text:
                return False, None

            # First line: two integers, e.g. "3 5"
            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2:
                return False, None

            start, end = int(lines_to_edit[0]), int(lines_to_edit[1])
            if len(text[1:]) == 0:
                return False, None

            return success, (start, end, codelines, text[1:], datasetcode)
        except Exception:
            return False, (None, None, None, None, None)


# ------------------------------------------------------------------
# SCORING FUNCTION
# ------------------------------------------------------------------
def get_score(outlined_plan, code, code_return, REWARD_MODEL_LLM, attempts=3, openai_api_key=None):
    """
    Query a 'reward model' to get a float score [0..1] representing
    how well the code followed the plan and produced the correct output.
    """
    error_str = ""
    for attempt in range(attempts):
        try:
            # Create system prompt
            system_text = (
                "You are a professor agent (expert reward model) evaluating:\n"
                "1) Adherence to the plan\n"
                "2) Quality of the code\n"
                "3) Correctness of the output\n"
                "Your score must be a float in [0,1].\n"
                'Respond in the format: ```SCORE\n<score>\n```'
            )
            scoring = query_model(
                model_str=REWARD_MODEL_LLM,
                system_prompt=system_text,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Plan:\n{outlined_plan}\n\n"
                    f"Produced Code:\n{code}\n\n"
                    f"Code Output:\n{code_return}\n"
                ),
                temp=0.6
            )
            performance = extract_prompt(scoring, "SCORE")
            performance = float(performance)
            return performance, f"The performance of your submission is: {performance}", True
        except Exception as e:
            error_str = str(e)
            return None, error_str, False
    return 0, error_str, False


# ------------------------------------------------------------------
# CODE REPAIR FUNCTION
# ------------------------------------------------------------------
def code_repair(code, error, ctype, REPAIR_LLM, openai_api_key=None):
    """
    Automated code repair attempts, returning either an EDIT command or REPLACE block
    based on the ctype parameter.
    """
    if ctype == "replace":
        repair_system_msg = (
            "You are an automated code repair tool.\n"
            "Goal: fix the code so the same error does not recur. Keep the code's behavior.\n"
            "Output MUST be wrapped in:\n```python\n<code>\n```\n"
            "Don't forget the opening '```python' and closing '```'."
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=REPAIR_LLM,
            system_prompt=repair_system_msg,
            prompt=f"Error:\n{error}\n\nCode:\n{code}",
            temp=0.8
        )
        return extract_prompt(model_resp, "python")

    elif ctype == "edit":
        repair_system_msg = (
            "You are an automated code repair tool.\n"
            "Goal: fix the code so the same error does not recur, while preserving original behavior.\n"
            "Output MUST use the CODE EDITING TOOL in the format:\n"
            "```EDIT N M\n<new lines>\n```"
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=REPAIR_LLM,
            system_prompt=repair_system_msg,
            prompt=f"Error:\n{error}\n\nCode:\n{code}",
            temp=0.2
        )
        return model_resp


# ------------------------------------------------------------------
# PRIMARY MLESolver CLASS
# ------------------------------------------------------------------
class MLESolver:
    """
    Orchestrates the code generation and editing process
    to solve a machine learning research plan.
    """
    def __init__(self, dataset_code, openai_api_key=None, notes=None,
                 max_steps=10, insights=None, plan=None, llm_str=None):
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

    # ----------------------
    # INITIAL SOLVE
    # ----------------------
    def initial_solve(self):
        """
        Initialize solver with an initial code generation (REPLACE).
        """
        self.best_score = None
        self.commands = [Replace()]
        self.model = self.llm_str

        # Generate initial code & store as best
        init_code, init_return, self.best_score = self.gen_initial_code()
        self.best_codes = [(copy(init_code), self.best_score, init_return)]

        # Switch to having both Edit + Replace commands
        self.code_lines = init_code
        self.model = self.llm_str
        self.commands = [Edit(), Replace()]
        self.prev_working_code = copy(self.code_lines)

    # ----------------------
    # GENERATE INITIAL CODE
    # ----------------------
    def gen_initial_code(self):
        num_attempts = 0
        error_hist = []

        while True:
            if num_attempts == 0:
                err_hist = ""
            else:
                err = (
                    f"Previous attempt failed with errors. Avoid repeating them.\n"
                    f"Error history:\n{error_hist}\n"
                )
                err_hist = err

            # Ask model to produce a REPLACE block
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=(
                    f"{err_hist}\n"
                    "Use ```REPLACE to create the initial code.\n"
                    "Please enter the ```REPLACE command below:\n"
                ),
                temp=1.0
            )

            model_resp = self.clean_text(model_resp)
            cmd_str, code_lines, prev_code_ret, exec_code, score = self.process_command(model_resp)

            print(f"@@@ INIT ATTEMPT: {num_attempts} --> {cmd_str}")
            print(f"$$$ Score: {score}")

            if score is not None:
                # Successfully generated & scored
                break

            error_hist.append(cmd_str)
            if len(error_hist) > 5:
                error_hist.pop(0)

            num_attempts += 1

        return code_lines, prev_code_ret, score

    # ----------------------
    # MAIN SOLVE LOOP
    # ----------------------
    def solve(self):
        num_attempts = 0
        best_pkg = None
        top_score = None

        self.prev_code_ret = None
        self.should_execute_code = False

        while True:
            # Encourage the user to produce EDIT or REPLACE
            if len(self.commands) == 2:
                cmd_app_str = "You must output either the ```EDIT or ```REPLACE command immediately. "
            else:
                cmd_app_str = ""

            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=(
                    f"The following is your history:\n{self.history_str()}\n\n"
                    f"{cmd_app_str}Now please enter a command:\n"
                ),
                temp=1.0
            )
            model_resp = self.clean_text(model_resp)

            # Temporarily revert code to a known best
            self.code_lines = copy(random.choice(self.best_codes)[0])

            cmd_str, code_lines, prev_code_ret, exec_code, score = self.process_command(model_resp)
            self.st_history.append([model_resp, prev_code_ret, code_lines, cmd_str])

            # Trim old history
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

        # Final best
        self.code_lines, self.prev_code_ret, self.should_execute_code, model_resp, cmd_str = best_pkg

        # If top scoring code is better than the last best code, store it
        if top_score is not None and (not self.best_codes or top_score > self.best_codes[-1][1]):
            if len(self.best_codes) >= self.max_codes:
                self.best_codes.pop(-1)
                self.code_reflect = self.reflect_code()
            self.best_codes.append((copy(self.code_lines), copy(top_score), self.prev_code_ret))
            self.best_codes.sort(key=lambda x: x[1], reverse=True)

        return model_resp, cmd_str

    # ----------------------
    # CODE REFLECTION
    # ----------------------
    def reflect_code(self):
        """
        Use the language model to reflect on previous best codes for improvements.
        """
        code_strs = ""
        sep_line = f"{'$' * 40}\n\n"
        for code_info in self.best_codes:
            c_lines, sc, ret = code_info
            snippet = self.generate_code_lines(c_lines)
            code_strs += f"{snippet}\nCode Return {sc}\n{sep_line}"

        prompt_text = (
            "Please reflect on the following sets of code. "
            "Propose improvements and rationale for higher performance on the benchmark."
        )
        system_msg = self.system_prompt(commands=False) + code_strs

        return query_model(
            prompt=prompt_text,
            system_prompt=system_msg,
            model_str=self.llm_str,
            openai_api_key=self.openai_api_key
        )

    # ----------------------
    # PROCESS COMMAND
    # ----------------------
    def process_command(self, model_resp):
        """
        Execute the command (EDIT or REPLACE) if found. Returns a tuple of:
            (cmd_str, code_lines, prev_code_ret, should_execute_code, score)
        """
        prev_code_ret = self.prev_code_ret
        should_execute_code = self.should_execute_code
        code_lines = copy(self.code_lines)

        remove_figures()

        with suppress_stdout():
            for cmd in self.commands:
                if cmd.matches_command(model_resp):
                    # -------------- CODE-EDIT --------------
                    if cmd.cmd_type == "CODE-edit":
                        score = None
                        failed = True
                        code_err = ""

                        for _try in range(GLOBAL_REPAIR_ATTEMPTS):
                            success, args = cmd.parse_command(model_resp, copy(self.code_lines), self.dataset_code)
                            if success:
                                cmd_return = cmd.execute_command(args)
                                code_err = f"Return from executing code: {cmd_return[2]}"
                                if cmd_return[0]:  # success
                                    code_lines = copy(cmd_return[1])
                                    # Evaluate code
                                    score, cmd_str, is_valid = get_score(
                                        self.plan,
                                        "\n".join(code_lines),
                                        cmd_return[2],
                                        openai_api_key=self.openai_api_key,
                                        REWARD_MODEL_LLM=self.llm_str
                                    )
                                    if is_valid:
                                        failed = False
                                        break
                                    code_err += f"\n[Score Response] {cmd_str}"

                            # Attempt automatic repair if it fails
                            repaired_code = code_repair(model_resp, code_err,
                                                        REPAIR_LLM=self.llm_str,
                                                        ctype="edit",
                                                        openai_api_key=self.openai_api_key)
                            model_resp = repaired_code
                            print(f"     * Attempting repair (EDIT) // try {_try} *")

                        if failed:
                            cmd_str = f"Code editing FAILED: {code_err}. Reverted to previous state."
                            print("$$$$ CODE EDIT (failed)")
                        else:
                            cmd_str = "Code was successfully edited."
                            prev_code_ret = copy(cmd_return[2])
                            should_execute_code = True
                            print("$$$$ CODE EDIT (success)")

                        return cmd_str, code_lines, prev_code_ret, should_execute_code, score

                    # -------------- CODE-REPLACE --------------
                    elif cmd.cmd_type == "CODE-replace":
                        score = None
                        failed = True
                        code_err = ""

                        for _try in range(GLOBAL_REPAIR_ATTEMPTS):
                            success, args = cmd.parse_command(model_resp, self.dataset_code)
                            code_err = f"Return from executing code: {args[1]}"

                            if success:
                                code_lines = copy(args[0])
                                score, cmd_str, is_valid = get_score(
                                    self.plan,
                                    "\n".join(code_lines),
                                    args[1],
                                    openai_api_key=self.openai_api_key,
                                    REWARD_MODEL_LLM=self.llm_str
                                )
                                if is_valid:
                                    failed = False
                                    break
                                code_err += f"\n[Score Response] {cmd_str}"

                            # Attempt automatic repair
                            replaced_code = extract_prompt(model_resp, "REPLACE")
                            repaired_code = code_repair(
                                replaced_code,
                                code_err,
                                ctype="replace",
                                openai_api_key=self.openai_api_key,
                                REPAIR_LLM=self.llm_str
                            )
                            model_resp = f"```REPLACE\n{repaired_code}\n```"
                            print(f"     * Attempting repair (REPLACE) // try {_try} *")

                        if failed:
                            cmd_str = (
                                f"Code replacement FAILED: {code_err}. "
                                "Reverted to previous state."
                            )
                            print("$$$$ CODE REPLACE (failed)")
                        else:
                            cmd_str = "Code was successfully replaced."
                            prev_code_ret = copy(args[1])
                            should_execute_code = True
                            print("$$$$ CODE REPLACE (success)")

                        return cmd_str, code_lines, prev_code_ret, should_execute_code, score

            # No recognized command
            print("$$$$ INVALID COMMAND (failed)")
            return "Command not supported, choose from existing commands", None, None, None, None

    # ----------------------
    # CLEAN TEXT
    # ----------------------
    @staticmethod
    def clean_text(text):
        """
        Convert code blocks from ```python\n to ```REPLACE\n, for consistency.
        """
        text = text.replace("```\n", "```")
        text = text.replace("```python\n", "```REPLACE\n")
        return text

    # ----------------------
    # HISTORY STRING
    # ----------------------
    def history_str(self):
        """
        Nicely format the solver's short-term history.
        """
        hist_str = ""
        for idx in range(len(self.st_history)):
            item = self.st_history[idx]
            model_resp, code_ret, lines, cmd_str = item
            hist_str += f"-------- History ({len(self.st_history)-idx} steps ago) -----\n"
            if model_resp:
                hist_str += f"LM Response:\n{model_resp}\n"
            hist_str += f"COMMAND Output:\n{cmd_str}\n"
            hist_str += f"Code:\n{'#'*20}\n{lines}\n{'#'*20}\n\n"
            hist_str += f"Feedback/Reflection:\n{code_ret}\n"
            hist_str += f"-------- End of history ({len(self.st_history)-idx} steps ago) -------\n\n"
        return hist_str

    # ----------------------
    # SYSTEM PROMPT
    # ----------------------
    def system_prompt(self, commands=True):
        """
        Builds the 'system' message for the language model, describing the role, instructions, etc.
        """
        result = (
            f"{self.role_description()}\n"
            f"The following are your task instructions: {self.phase_prompt()}\n"
            f"Provided below are insights from a literature review:\n{self.insights}\n"
            f"{self.code_reflect}\n"
            f"The following are notes/tips:\n{self.notes}\n"
            f"The plan:\n{self.plan}\n"
            f"{self.generate_dataset_descr_prompt()}\n"
            "You should generate at least two figures (Figure_1.png, Figure_2.png).\n"
            "Your method must not yield 0% accuracy.\n"
            "Use print statements to explain results in detail.\n"
        )

        if commands:
            result += (
                f"The following are commands you can use: {self.command_descriptions()}\n"
                "Please do not execute multiple commands at once.\n"
            )
        return result

    # ----------------------
    # GENERATE CODE LINES
    # ----------------------
    @staticmethod
    def generate_code_lines(code):
        """
        Return line-numbered code for printing & reflection.
        """
        return "\n".join(f"{i} |{line}" for i, line in enumerate(code))

    # ----------------------
    # FEEDBACK
    # ----------------------
    def feedback(self, code_return):
        """
        Provide feedback after code execution. If error, reflect on how to fix it.
        If successful, reflect on improvements.
        """
        if code_return:
            code_str = self.generate_code_lines(self.code_lines)

            if "[CODE EXECUTION ERROR]" in code_return:
                # Reflection prompt for error
                reflect_prompt = (
                    f"Your code returned this error:\n{code_return}\n\n"
                    f"{code_str}\n"
                    "Please reflect on why this error occurred and propose line-by-line fixes."
                )
            elif os.path.exists("submission.csv"):
                # If code produced a submission
                self.prev_working_code = copy(self.code_lines)
                grade_return, _, _ = get_score(
                    self.plan, "\n".join(self.prev_working_code),
                    code_return, openai_api_key=self.openai_api_key,
                    REWARD_MODEL_LLM=self.llm_str
                )
                reflect_prompt = (
                    f"Your code submitted successfully with score {grade_return}.\n"
                    f"{code_str}\n\n"
                    "Reflect on further improvements (hyperparameters, data augmentation, etc.)."
                )
                # Clean up leftover CSV if any
                for file in os.listdir("."):
                    if file.endswith(".csv"):
                        os.system(f"rm {file}")
            else:
                # No error, but also no submission
                reflect_prompt = (
                    "Your code ran without error but did not submit a file.\n"
                    f"{code_str}\n\n"
                    "Reflect on how to produce a valid submission next time."
                )
        else:
            code_return = "No changes were made to the code."
            reflect_prompt = "Reflect on future plans and next steps for improvement."

        reflection = self.reflection(reflect_prompt, code_return)
        return f"Code return: {code_return}\n\nReflection: {reflection}"

    # ----------------------
    # REFLECTION
    # ----------------------
    def reflection(self, reflect_prompt, code_return):
        """
        Reflection on how to fix or improve the code in the next iteration.
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

    # ----------------------
    # DATASET DESCRIPTION
    # ----------------------
    def generate_dataset_descr_prompt(self):
        """
        The dataset code is added automatically, so the user doesn't rewrite it.
        """
        return (
            f"The following dataset code is prepended automatically:\n{self.dataset_code}"
        )

    # ----------------------
    # PHASE PROMPT
    # ----------------------
    @staticmethod
    def phase_prompt():
        return (
            "You are an ML engineer writing code for a research project.\n"
            "Your code should run all experiments in the plan in a single file.\n"
            "You cannot pip install new libraries. Rely on pre-installed ones.\n"
            "Focus on correctness and clarity."
        )

    # ----------------------
    # ROLE DESCRIPTION
    # ----------------------
    @staticmethod
    def role_description():
        return (
            "You are an expert machine learning engineer at a top university, "
            "solving complex ML research challenges."
        )

    # ----------------------
    # COMMON CODE ERRORS
    # ----------------------
    @staticmethod
    def _common_code_errors():
        """
        Potential pitfalls and advice. 
        """
        return (
            "Make sure to import everything you use.\n"
            "Reflect to avoid bugs or compilation issues.\n"
            "Use commands EXACTLY (EDIT or REPLACE blocks). "
            "Never attempt multiple commands in one block.\n"
            "Avoid TensorFlow/Keras; prefer PyTorch or scikit-learn.\n"
        )

    # ----------------------
    # COMMAND DESCRIPTIONS
    # ----------------------
    def command_descriptions(self):
        cmd_strings = "\n".join([c.docstring() for c in self.commands])
        instructions = (
            "Commands must be in the form:\n"
            "```COMMAND\n<content>\n```\n"
            "where COMMAND is one of EDIT/REPLACE. Do not run multiple at once.\n"
        )
        return instructions + self._common_code_errors() + "\n" + cmd_strings

    # ----------------------
    # RUN CODE
    # ----------------------
    def run_code(self):
        """
        If no code changes have been made, return existing code's result;
        otherwise, re-execute code.
        """
        if self.prev_code_ret is not None:
            return self.prev_code_ret
        elif self.should_execute_code:
            return execute_code("\n".join(self.code_lines))
        return "No changes to execute."
