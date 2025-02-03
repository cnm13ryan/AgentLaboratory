import os
import sys
import random
import string
from copy import copy 
from contextlib import contextmanager
from abc import abstractmethod

from utils import *
from tools import *
from inference import *
from common_imports import *
from agents import get_score

@contextmanager
def suppress_stdout():
    """
    Context manager to temporarily redirect stdout to dev/null.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class Command:
    """
    Base abstract command class.
    """
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



"""
@@@@@@@@@@@@@@@@@@
@@ SEARCH TOOLS @@
@@@@@@@@@@@@@@@@@@
"""

class Arxiv(Command):
    def __init__(self):
        super().__init__()
        self.arxiv_eng = ArxivSearch()
        self.num_papers_per_search = 10
        self.cmd_type = "SEARCH-arxiv"

    def docstring(self) -> str:
        return (
            "============= ARXIV SEARCH TOOL ============="
            "You also have access to machine learning paper from Arxiv. "
            "To search for summaries of papers on arxiv you can use the following command:\n"
            "SUMMARY\n<search query>\n\n"
            "where <search query> is a string that will be used as the search query to find papers with semantically similar content.\n"
            "To get the full paper text for an arXiv paper, use the following command:\n"
            "FULL_TEXT\n<arxiv paper id>\n\n"
            "where <arxiv paper id> is the ID of the arXiv paper (from the SUMMARY command). "
            "Make sure to read the full text before adding it to your list of relevant papers.\n"
            "Take note of techniques, hyperparameters, and implementation details found in the papers."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> command ("SUMMARY" or "FULL_TEXT")
        # args[1] -> query or paper ID
        if args[0] == "SUMMARY":
            return self.arxiv_eng.find_papers_by_str(args[1], self.num_papers_per_search)
        elif args[0] == "FULL_TEXT":
            return self.arxiv_eng.retrieve_full_paper_text(args[1])
        raise Exception("Invalid Arxiv Search")

    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the cmd_str is a SUMMARY or FULL_TEXT command.
        """
        if "\nSUMMARY" in cmd_str:
            return True
        elif "\nFULL_TEXT" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        """
        Parses the command to extract the search query or the paper ID.
        Returns (success_bool, data_tuple).
        """
        sum_text = extract_prompt(args[0], "SUMMARY").split("\n")
        full_text = extract_prompt(args[0], "FULL_TEXT").split("\n")

        # If neither summary nor full_text found, fail
        if len(sum_text) == 0 and len(full_text) == 0:
            return False, None

        # If we found a SUMMARY command
        if len(sum_text) > 0:
            return True, ("SUMMARY", sum_text,)

        # If we found a FULL_TEXT command
        if len(full_text) > 0:
            # BUGFIX: Use full_text instead of sum_text
            return True, ("FULL_TEXT", full_text,)


"""
@@@@@@@@@@@@@@@@@@@
@@ WRITING TOOLS @@
@@@@@@@@@@@@@@@@@@@
"""

class PaperReplace(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "PAPER-replace"

    def docstring(self) -> str:
        return (
            "============= PAPER REPLACING TOOL =============\n"
            "Allows you to entirely re-write/replace all of the current LaTeX text.\n"
            "Command usage:\n\n"
            "REPLACE\n<latex here>\n\n"
            "The new latex will replace the old entirely. It is tested before finalizing."
        )

    def execute_command(self, *args) -> str:
        # args[0][0] -> new latex
        new_latex = args[0][0]
        return new_latex

    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the cmd_str contains REPLACE command.
        """
        return "\nREPLACE" in cmd_str

    def parse_command(self, *args) -> tuple:
        """
        Parses the REPLACE command and compiles the new latex.
        Returns (success_bool, (list_of_lines, latex_ret)).
        """
        new_latex = extract_prompt(args[0], "REPLACE")
        latex_ret = compile_latex(new_latex, compile=args[1])
        if "[CODE EXECUTION ERROR]" in latex_ret:
            return False, (None, latex_ret,)
        return True, (new_latex.split("\n"), latex_ret)


class PaperEdit(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "PAPER-edit"

    def docstring(self) -> str:
        return (
            "============= PAPER EDITING TOOL =============\n"
            "Command usage:\n\n"
            "EDIT N M\n<new lines>\n\n"
            "Lines N through M (inclusive) of the LaTeX are removed and replaced by the <new lines>. "
            "Tested before finalizing."
        )

    def execute_command(self, *args) -> str:
        """
        Perform the actual editing of lines N through M.
        Returns a tuple of (success_bool, updated_latex_lines, compile_result).
        """
        try:
            # args[0] is a tuple containing (N, M, current_latex_lines, lines_to_add, compile_flag)
            command_args = args[0]
            n, m = command_args[0], command_args[1]
            current_latex = command_args[2]
            lines_to_add = list(reversed(command_args[3]))
            lines_to_replace = list(reversed(range(n, m + 1)))

            # Remove lines in [n..m]
            for ln in lines_to_replace:
                current_latex.pop(ln)

            # Insert new lines
            for line in lines_to_add:
                current_latex.insert(n, line)

            new_latex = "\n".join(current_latex)
            latex_ret = compile_latex(new_latex, compile=command_args[4])

            if "error" in latex_ret.lower():
                return (False, None, latex_ret)
            return (True, current_latex, latex_ret)

        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        """
        Checks if the cmd_str contains EDIT command.
        """
        return "\nEDIT" in cmd_str

    def parse_command(self, *args) -> tuple:
        """
        Extracts the command, line indices, and new lines from the EDIT command.
        Returns (success_bool, (n, m, current_latex, lines_to_add)).
        """
        cmd_str, latexlines = args[0], args[1]
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0:
                return False, (None, None, None, None)

            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2:
                return False, (None, None, None, None)
            n, m = int(lines_to_edit[0]), int(lines_to_edit[1])

            if len(text[1:]) == 0:
                return False, (None, None, None, None)

            return True, (n, m, latexlines, text[1:])
        except Exception:
            return False, (None, None, None, None)


# Tips used by PaperSolver
per_section_tips = {
    "abstract": """
- TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard?
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- Must be a single paragraph, no breaks.
""",
    "introduction": """
- Longer version of the Abstract
- What are we trying to do and why is it relevant?
- Why is this hard?
- How do we solve it (our contribution!)
- How do we verify that we solved it (Experiments and results)
- List contributions as bullet points
- Optional: future work
""",
    "related work": """
- Compare and contrast with existing attempts.
- Must explain why or why not relevant to our approach.
""",
    "background": """
- Concepts and prior work needed to understand our method.
- Possibly includes problem setting, notation, assumptions.
""",
    "methods": """
- Detailed description of what we do and why.
- Provide math equations where needed.
""",
    "experimental setup": """
- How do we test our method? Datasets, metrics, hyperparams, etc.
""",
    "results": """
- Show results from the method. No fabricated data.
- Compare with baselines if available.
- Discuss limitations.
""",
    "discussion": """
- Summarize the entire paper.
- Possibly mention future work or open questions.
""",
}

class PaperSolver:
    """
    Orchestrates the creation/editing of a LaTeX paper using a language model and commands.
    """
    def __init__(
        self,
        llm_str,
        notes=None,
        max_steps=10,
        insights=None,
        plan=None,
        exp_code=None,
        exp_results=None,
        lit_review=None,
        ref_papers=None,
        topic=None,
        openai_api_key=None,
        compile_pdf=True
    ):
        # Defaulting optional params to empty or sensible defaults
        self.notes = notes if notes else []
        self.plan = plan if plan else ""
        self.exp_code = exp_code if exp_code else ""
        self.exp_results = exp_results if exp_results else ""
        self.lit_review = lit_review if lit_review else ""
        self.insights = insights if insights else ""
        self.ref_papers = ref_papers if ref_papers else ""
        self.topic = topic if topic else ""

        self.compile_pdf = compile_pdf
        self.llm_str = llm_str
        self.max_papers = 1
        self.st_hist_len = 10
        self.min_gen_trials = 2
        self.max_steps = max_steps

        self.paper_lines = str()
        self.prev_paper_ret = str()
        self.section_related_work = {}
        self.openai_api_key = openai_api_key

    def solve(self):
        """
        Main solver loop—queries a language model for commands and executes them
        until a high scoring version is found or max attempts are reached.
        """
        num_attempts = 0
        best_pkg = None
        top_score = None
        self.prev_paper_ret = None

        while True:
            self.paper_lines = copy(random.choice(self.best_report)[0])
            model_resp = query_model(
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt="\nNow please enter a command: ",
                temp=1.0,
                openai_api_key=self.openai_api_key
            )
            model_resp = self.clean_text(model_resp)
            cmd_str, paper_lines, prev_paper_ret, score = self.process_command(model_resp)

            if score is not None:
                if top_score is None:
                    best_pkg = copy(paper_lines), copy(prev_paper_ret), copy(model_resp), copy(cmd_str)
                    top_score = score
                elif score > top_score:
                    best_pkg = copy(paper_lines), copy(prev_paper_ret), copy(model_resp), copy(cmd_str)
                    top_score = score

            if num_attempts >= self.min_gen_trials and top_score is not None:
                break

            print(f"@@@ Command Exec // Attempt {num_attempts}: {cmd_str}")
            print(f"$$$ Score: {score}")
            num_attempts += 1

        # Apply the best found
        self.paper_lines, self.prev_paper_ret, model_resp, cmd_str = best_pkg

        # Insert top scoring paper
        if top_score > self.best_report[-1][1]:
            if len(self.best_report) >= self.max_papers:
                self.best_report.pop(-1)
            self.best_report.append((copy(self.paper_lines), copy(top_score), self.prev_paper_ret))
            self.best_report.sort(key=lambda x: x[1], reverse=True)

        return model_resp, cmd_str

    def initial_solve(self):
        """
        Initialize the solver with a first pass of scaffolding commands.
        """
        self.best_score = None
        self.commands = [PaperReplace()]
        self.model = f"{self.llm_str}"

        init_report, init_return, self.best_score = self.gen_initial_report()
        self.best_report = [(copy(init_report), self.best_score, init_return) for _ in range(1)]

        self.paper_lines = init_report
        self.model = f"{self.llm_str}"
        self.commands = [PaperEdit()]
        self.prev_working_report = copy(self.paper_lines)

    @staticmethod
    def clean_text(text):
        """
        Replaces certain newlines to keep the formatting consistent.
        """
        return text.replace("\n\n", "\n")

    def gen_initial_report(self):
        """
        Creates an initial scaffold for the paper, section by section.
        """
        num_attempts = 0
        arx = ArxivSearch()
        section_scaffold = ""

        for _section in ["scaffold", "abstract", "introduction", "related work",
                         "background", "methods", "experimental setup", "results", "discussion"]:
            section_complete = False
            if _section in ["introduction", "related work", "background", "methods", "discussion"]:
                attempts = 0
                papers = ""
                first_attempt = True
                while len(papers) == 0:
                    if attempts > 5:
                        break
                    att_str = ""
                    if not first_attempt:
                        att_str = "This is not your first attempt. Try a simpler query."
                    search_query = query_model(
                        model_str=f"{self.llm_str}",
                        prompt=(
                            f"Given the research topic {self.topic} and plan:\n{self.plan}\n"
                            "Come up with an arXiv search query.\n"
                            "Respond with only the query text."
                            f"{att_str}"
                        ),
                        system_prompt=f"You are a research paper finder for section {_section}.",
                        openai_api_key=self.openai_api_key
                    )
                    search_query = search_query.replace('"', '')
                    papers = arx.find_papers_by_str(query=search_query, N=10)
                    first_attempt = False
                    attempts += 1

                if len(papers) != 0:
                    self.section_related_work[_section] = papers

            while not section_complete:
                section_scaffold_temp = copy(section_scaffold)
                if num_attempts == 0:
                    err = ""
                else:
                    err = f"Previous command: {model_resp} // Error: {cmd_str}"

                if _section == "scaffold":
                    prompt = f"{err}\nNow please enter the \nREPLACE command to create the scaffold:\n"
                else:
                    rp = ""
                    if _section in self.section_related_work:
                        rp = f"Here are papers to cite: {self.section_related_work[_section]}.\n"
                    prompt = (
                        f"{err}\n{rp}\nNow please enter the \nREPLACE command to create "
                        f"the section '{_section}', text only (no packages or section titles):\n"
                    )

                model_resp = query_model(
                    model_str=self.model,
                    system_prompt=self.system_prompt(section=_section),
                    prompt=prompt,
                    temp=0.8,
                    openai_api_key=self.openai_api_key
                )
                model_resp = self.clean_text(model_resp)

                if _section == "scaffold":
                    # minimal check for placeholders
                    for _sect in ["[ABSTRACT HERE]", "[INTRODUCTION HERE]",
                                  "[METHODS HERE]", "[RESULTS HERE]", "[DISCUSSION HERE]"]:
                        if _sect not in model_resp:
                            cmd_str = f"Error: missing placeholder '{_sect}'."
                            print("@@@ INIT ATTEMPT:", cmd_str)
                            continue
                elif _section != "scaffold":
                    new_text = extract_prompt(model_resp, "REPLACE")
                    section_scaffold_temp = section_scaffold_temp.replace(
                        f"[{_section.upper()} HERE]", new_text
                    )
                    model_resp = f"\nREPLACE\n{copy(section_scaffold_temp)}\n"

                cmd_str, latex_lines, prev_latex_ret, score = self.process_command(
                    model_resp, scoring=False
                )
                print(f"@@@ INIT ATTEMPT // Attempt {num_attempts}: {cmd_str}")

                if score is not None:
                    section_complete = True
                    section_scaffold = "\n".join(latex_lines)
                num_attempts += 1

            self.paper_lines = section_scaffold.split("\n")
            print("########## SCAFFOLD", _section.upper(), "CREATED ##########")

        print("########## FULL SCAFFOLD CREATED ##########")
        return latex_lines, prev_latex_ret, score

    def process_command(self, model_resp, scoring=True):
        """
        Executes recognized commands (PaperEdit, PaperReplace) on the LaTeX text,
        then returns updated lines & possible 'score'.
        """
        cmd_str = None
        score = None
        prev_paper_ret = self.prev_paper_ret
        paper_lines = copy(self.paper_lines)

        # Adjust paths for images if present
        if "\\includegraphics[width=\\textwidth]{Figure_1.png}" in model_resp \
           or "\\includegraphics[width=\\textwidth]{Figure_2.png}" in model_resp:
            cwd = os.getcwd()
            model_resp = model_resp.replace(
                "\\includegraphics[width=\\textwidth]{Figure_1.png}",
                f"\\includegraphics[width=\\textwidth]{{{cwd}/Figure_1.png}}"
            )
            model_resp = model_resp.replace(
                "\\includegraphics[width=\\textwidth]{Figure_2.png}",
                f"\\includegraphics[width=\\textwidth]{{{cwd}/Figure_2.png}}"
            )

        for cmd in self.commands:
            if cmd.matches_command(model_resp):
                # PAPER-EDIT
                if cmd.cmd_type == "PAPER-edit":
                    score = None
                    failed = True
                    success, parse_args = cmd.parse_command(model_resp, paper_lines)
                    if success:
                        # parse_args = (n, m, current_latex, lines_to_add)
                        exec_args = (parse_args[0], parse_args[1], paper_lines,
                                     parse_args[3], self.compile_pdf)
                        result = cmd.execute_command(exec_args)
                        success = success and result[0]
                        paper_err = ""
                        if success:
                            paper_lines = copy(result[1])
                            if scoring:
                                score, cmd_str, is_valid = get_score(
                                    self.plan, "\n".join(paper_lines), reward_model_llm=self.llm_str
                                )
                            else:
                                score, cmd_str, is_valid = 0.0, "Paper scored successfully", True

                            if is_valid:
                                failed = False
                            paper_err += f"\nReturn from latex: {cmd_str}"

                        if failed:
                            cmd_str = (
                                f"Paper edit FAILED: {paper_err}. Reverting to original."
                            )
                            print("$$$$ PAPER EDIT (failed)")
                        else:
                            cmd_str = "Paper edit succeeded."
                            prev_paper_ret = copy(result[2])
                            print("$$$$ PAPER EDIT (success)")

                # PAPER-REPLACE
                elif cmd.cmd_type == "PAPER-replace":
                    score = None
                    failed = True
                    success, parse_args = cmd.parse_command(model_resp, self.compile_pdf)
                    paper_err = ""
                    if success:
                        new_lines, latex_ret = parse_args
                        if scoring:
                            score, cmd_str, is_valid = get_score(
                                self.plan, "\n".join(new_lines), reward_model_llm=self.llm_str
                            )
                        else:
                            score, cmd_str, is_valid = 0.0, "Paper scored successfully", True

                        if is_valid:
                            failed = False
                        paper_err += f"\nReturn from latex: {cmd_str}"
                        if not failed:
                            paper_lines = copy(new_lines)
                            prev_paper_ret = copy(latex_ret)

                    if failed:
                        cmd_str = (
                            f"Paper replacement FAILED: {paper_err}. Reverting to original."
                        )
                        print("$$$$ PAPER REPLACE (failed)")
                    else:
                        cmd_str = "Paper replacement succeeded."
                        print("$$$$ PAPER REPLACE (success)")

        return cmd_str, paper_lines, prev_paper_ret, score

    def generate_paper_lines(self, code):
        """
        Returns code lines with line numbers for clarity in editing.
        """
        lines_str = ""
        for index, line in enumerate(code):
            lines_str += f"{index} | {line}\n"
        return lines_str

    def system_prompt(self, commands=True, section=None):
        """
        Builds a system prompt for the model, possibly focusing on a specific section.
        """
        if section == "abstract":
            length_req = "This section should be ONLY 1 paragraph."
        else:
            length_req = (
                "This section should be ~2-4 paragraphs with multiple lines of text."
            )

        methods_str = ""
        if section == "methods":
            fig1_text = (
                "\n\\begin{figure}[h]\n"
                "\\caption{<caption here>}\n"
                "\\centering\n"
                "\\includegraphics[width=\\textwidth]{Figure_1.png}\n"
                "\\label{fig:fig1}\n"
                "\\end{figure}\n"
            )
            fig2_text = (
                "\n\\begin{figure}[h]\n"
                "\\caption{<caption here>}\n"
                "\\centering\n"
                "\\includegraphics[width=\\textwidth]{Figure_2.png}\n"
                "\\label{fig:fig1}\n"
                "\\end{figure}\n"
            )
            if os.path.exists("Figure_1.png") and os.path.exists("Figure_2.png"):
                methods_str += (
                    "You must include Figure_1.png and Figure_2.png:\n"
                    f"{fig1_text}\n{fig2_text}"
                )
            elif os.path.exists("Figure_1.png"):
                methods_str += f"Include Figure_1.png:\n{fig1_text}"
            elif os.path.exists("Figure_2.png"):
                methods_str += f"Include Figure_2.png:\n{fig2_text}"

        section_cmd = ""
        if section is not None:
            if section == "scaffold":
                section_cmd = (
                    "Your objective: build the scaffolding for the paper. "
                    "No text in the body, just placeholders like [ABSTRACT HERE], [INTRODUCTION HERE]... "
                    "Your paper should have 8 sections total: Abstract, Introduction, Background, Related Work, "
                    "Methods, Experimental Setup, Results, Discussion."
                )
            else:
                section_cmd = (
                    f"Generate latex only for the '{section}' section. {length_req}\n"
                    "Do not include packages or \\section commands. "
                    "No title or date. Insert text only.\n"
                    f"Here are tips:\n{per_section_tips[section]}\n{methods_str}\n"
                )

        paper_len = sum(
            token.strip(string.punctuation).isalpha()
            for token in ("".join(self.paper_lines)).split()
        )
        if paper_len < 4000:
            paper_progress = f"We currently have {paper_len} words; we need {4000 - paper_len} more."
        else:
            paper_progress = ""

        cmd_set = ""
        if commands:
            cmd_set = f"The following commands are available:\n{self.command_descriptions()}"

        if self.ref_papers:
            ref_papers_header = "Here is a high-quality reference paper:\n" + "\n".join(self.ref_papers) + "\n"
        else:
            ref_papers_header = ""

        lit_snippet = self.lit_review[:20000]

        return (
            f"{ref_papers_header}"
            f"{self.role_description()}\n\n"
            "Task instructions:\n"
            f"{self.phase_prompt()}\n\n"
            f"Notes:\n{self.notes}\n\n"
            f"Literature review:\n{lit_snippet}\n\n"
            f"Plan:\n{self.plan}\n\n"
            f"Code:\n{self.exp_code}\n\n"
            f"Results:\n{self.exp_results}\n\n"
            f"Insights:\n{self.insights}\n\n"
            "Style: objective and straightforward.\n"
            f"The paper should be around 4000 words total. {paper_progress}\n\n"
            f"{cmd_set}\n\n"
            "Current paper:\n"
            f"{self.generate_paper_lines(self.paper_lines)}\n\n"
            f"{section_cmd}"
        )

    def command_descriptions(self):
        """
        Returns docstrings of the commands for the system prompt.
        """
        return "\n".join([cmd.docstring() for cmd in self.commands])

    def role_description(self):
        return (
            "You are a CS PhD student writing a paper for the ICLR conference. "
            "Your paper has exactly 8 sections: 1) Abstract, 2) Introduction, 3) Background, "
            "4) Related Work, 5) Methods, 6) Experimental Setup, 7) Results, 8) Discussion."
        )

    def phase_prompt(self):
        return (
            "You have submitted a paper to ICLR. Your goal is to refine this paper "
            "so that it is accepted by reviewers. The paper must be about 8 pages or ~4000 words."
        )
