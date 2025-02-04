from utils import *
from tools import *
from inference import *


def extract_json_between_markers(llm_output):
    # Guard clause for empty output:
    if not llm_output:
        return None

    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found

def _build_reviewer_system_prompt(reviewer_type=""):
    template_instructions = """
    Respond in the following format:

    THOUGHT:
    <THOUGHT>

    REVIEW JSON:
    ```json
    <JSON>
    ```

    In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
    In <JSON>, provide the review in JSON format with the following fields in the order:
    - "Summary"
    - "Strengths"
    - "Weaknesses"
    - "Originality"
    - "Quality"
    - "Clarity"
    - "Significance"
    - "Questions"
    - "Limitations"
    - "Ethical Concerns"
    - "Soundness"
    - "Presentation"
    - "Contribution"
    - "Overall"
    - "Confidence"
    - "Decision"

    Decision must be only "Accept" or "Reject".
    """
    neurips_form = (
        """
        ## Review Form
        (NeurIPS style questions and guidelines)
        
        1. Summary, Strengths, Weaknesses
        2. Questions
        3. Limitations
        4. Ethical concerns
        5. Soundness (1-4)
        6. Presentation (1-4)
        7. Contribution (1-4)
        8. Overall (1-10)
        9. Confidence (1-5)
        
        You must make sure that all sections are properly created.
        """ + template_instructions
    )
    base_str = (
        "You are an AI researcher who is reviewing a paper that was submitted "
        "to a prestigious ML venue. Be critical and cautious."
    )
    return f"{base_str} {reviewer_type}\n{neurips_form}"

def compute_performance_from_review(review_json):
    """
    Given review_json with fields like 'Overall', 'Soundness', 'Confidence', etc.,
    parse them into numeric values, apply weighting, and compute a final score.
    Returns the performance score as a float.
    """

    # Convert string-based fields into numeric (normalizing to 0..1 ranges)
    overall = int(review_json["Overall"]) / 10
    soundness = int(review_json["Soundness"]) / 4
    confidence = int(review_json["Confidence"]) / 5
    contribution = int(review_json["Contribution"]) / 4
    presentation = int(review_json["Presentation"]) / 4
    clarity = int(review_json["Clarity"]) / 4
    originality = int(review_json["Originality"]) / 4
    quality = int(review_json["Quality"]) / 4
    significance = int(review_json["Significance"]) / 4

    # Weighting factors for the final performance metric
    clarity_weight = 0.1
    quality_weight = 0.1
    overall_weight = 1.0
    soundness_weight = 0.1
    confidence_weight = 0.1
    originality_weight = 0.1
    significance_weight = 0.1
    contribution_weight = 0.4
    presentation_weight = 0.2

    # Calculate the sum of all weights
    max_score = (
        clarity_weight
        + quality_weight
        + overall_weight
        + soundness_weight
        + confidence_weight
        + originality_weight
        + significance_weight
        + contribution_weight
        + presentation_weight
    )

    # Weighted average, scaled to 10
    performance = (
        (
            (soundness_weight * soundness)
            + (presentation_weight * presentation)
            + (confidence_weight * confidence)
            + (contribution_weight * contribution)
            + (overall_weight * overall)
            + (originality_weight * originality)
            + (significance_weight * significance)
            + (clarity_weight * clarity)
            + (quality_weight * quality)
        )
        / max_score
    ) * 10

    return performance

def get_score(outlined_plan, latex, reward_model_llm, reviewer_type=None, attempts=3, openai_api_key=None):
    """
    Given a plan (outlined_plan) and associated LaTeX (latex),
    use a reward model LLM (reward_model_llm) to generate a review,
    parse it into JSON, and derive a performance score.
    Returns:
        performance (float): Calculated performance score.
        message (str): Explanation or error reason.
        success (bool): Whether scoring succeeded or not.
    """
    # Guard clauses for missing critical inputs:
    if not outlined_plan:
        return 0.0, "Missing 'outlined_plan'.", False
    if not latex:
        return 0.0, "Missing 'latex'.", False
    if not reward_model_llm:
        return 0.0, "Missing 'reward_model_llm' (LLM model name).", False

    exception_msg = ""
    for _attempt in range(attempts):
        try:
            final_reviewer_type = reviewer_type or ""
            sys = _build_reviewer_system_prompt(final_reviewer_type)

            # Query the model
            scoring = query_model(
                model_str=reward_model_llm,
                system_prompt=sys,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Outlined Plan: {outlined_plan}\n\n"
                    f"Latex of the submission: \n{latex}\n\n"
                ),
                temp=0.0
            )

            # Parse JSON from the LLM output
            review_json = extract_json_between_markers(scoring)
            if not review_json:
                return 0.0, "Could not parse valid JSON review.", False

            # Use the helper function to compute performance
            performance = compute_performance_from_review(review_json)

            # Return the final score, plus the model's generated text
            return (
                performance,
                f"The performance of your submission is: {performance}\n{scoring}",
                True
            )

        except Exception as ex:
            exception_msg = str(ex)
            # Try next attempt if any remain
            continue

    # If all attempts failed
    return 0.0, exception_msg, False


class ReviewersAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, openai_api_key=None):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.model = model
        self.openai_api_key = openai_api_key

    def inference(self, plan, report):
        reviewer_1 = "You are a harsh but fair reviewer and expect good experiments that lead to insights for the research topic."
        review_1 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_1, openai_api_key=self.openai_api_key)

        reviewer_2 = "You are a harsh and critical but fair reviewer who is looking for an idea that would be impactful in the field."
        review_2 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_2, openai_api_key=self.openai_api_key)

        reviewer_3 = "You are a harsh but fair open-minded reviewer that is looking for novel ideas that have not been proposed before."
        review_3 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_3, openai_api_key=self.openai_api_key)

        return f"Reviewer #1:\n{review_1}, \nReviewer #2:\n{review_2}, \nReviewer #3:\n{review_3}"


class BaseAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None):
        # Guard clauses:
        if model is None:
            raise ValueError("model must be specified (string).")
        if max_steps <= 0:
            raise ValueError("max_steps should be > 0.")

        if notes is None:
            self.notes = []
        else:
            self.notes = notes

        self.max_steps = max_steps
        self.model = model
        self.phases = []
        self.plan = str()
        self.report = str()
        self.history = list()
        self.prev_comm = str()
        self.prev_report = str()
        self.exp_results = str()
        self.dataset_code = str()
        self.results_code = str()
        self.lit_review_sum = str()
        self.interpretation = str()
        self.prev_exp_results = str()
        self.reviewer_response = str()
        self.prev_results_code = str()
        self.prev_interpretation = str()
        self.openai_api_key = openai_api_key

        self.second_round = False
        self.max_hist_len = 15

    def set_model_backbone(self, model):
        if not model:
            raise ValueError("model cannot be empty.")
        self.model = model

    @staticmethod
    def clean_text(text):
        """
        Fix minor corrections
        :return: (str) corrected text
        """
        text = text.replace("```\n", "```")
        return text

    def build_system_prompt(self, phase):
        """
        Helper that composes the system prompt from:
          1) role description
          2) phase prompt
          3) command descriptions
        """
        return (
            f"You are {self.role_description()}.\n"
            f"Task instructions: {self.phase_prompt(phase)}\n"
            f"{self.command_descriptions(phase)}"
        )

    def build_user_prompt(self, research_topic, phase, step, feedback):
        """
        Helper to build the user prompt portion (context, history, feedback, etc.).
        """
        if not research_topic:
            return "No research topic provided."
        if not phase:
            return "No phase provided."
        if step < 0:
            return "Step cannot be negative."

        context_str = self.context(phase)
        history_str = "\n".join([h[1] for h in self.history])
        complete_str = ""
        if step / (self.max_steps - 1) > 0.7:
            complete_str = "You must finish this task and submit as soon as possible!"

        prompt = (
            f"{context_str}\n"
            f"{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n"
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"[Objective] Your goal is to perform research on the following topic: {research_topic}\n"
            f"Feedback: {feedback}\n"
            f"Notes: {[n for n in self.notes if phase in n.get('phases', [])]}\n"
            f"Your previous command was: {self.prev_comm}.\n"
            f"Please produce a single command below:\n"
        )
        return prompt

    def inference(self, research_topic, phase, step, feedback="", temp=None):
        # Guard clauses:
        if not research_topic:
            return "No research topic provided."
        if not phase:
            return "No phase provided."
        if step < 0:
            return "Step cannot be negative."

        # Build system prompt
        sys_prompt = self.build_system_prompt(phase)
        
        # Build user prompt
        user_prompt = self.build_user_prompt(research_topic, phase, step, feedback)

        # Query the model
        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt,
            temp=temp,
            openai_api_key=self.openai_api_key
        )

        # Clean and store the response
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp

        # Handle expiration in feedback if present
        steps_exp = None
        if feedback and "```EXPIRATION" in feedback:
            try:
                steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
                feedback = extract_prompt(feedback, "EXPIRATION")
            except ValueError:
                pass

        # Append the current step to history and manage expiration
        self.history.append((steps_exp, f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {model_resp}"))
        for i in reversed(range(len(self.history))):
            if self.history[i][0] is not None:
                new_exp = self.history[i][0] - 1
                self.history[i] = (new_exp, self.history[i][1])
                if new_exp < 0:
                    self.history.pop(i)
        
        # Limit history length
        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)

        return model_resp

    def reset(self):
        self.history.clear()  # Clear the deque
        self.prev_comm = ""

    def context(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def phase_prompt(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def role_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def command_descriptions(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def example_command(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")


class ProfessorAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["report writing"]

    def generate_readme(self):
        sys_prompt = f"""You are {self.role_description()} \n Here is the written paper \n{self.report}. Task instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a readme.md for a github repository."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the readme below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp.replace("```markdown", "")

    def context(self, phase):
        #sr_str = str()
        #if self.second_round:
        #    sr_str = (
        #        f"The following are results from the previous experiments\n",
        #        f"Previous Experiment code: {self.prev_results_code}\n"
        #        f"Previous Results: {self.prev_exp_results}\n"
        #        f"Previous Interpretation of results: {self.prev_interpretation}\n"
        #        f"Previous Report: {self.prev_report}\n"
        #        f"{self.reviewer_response}\n\n\n"
        #    )
        #if phase == "report writing":
        #    return (
        #        sr_str,
        #        f"Current Literature Review: {self.lit_review_sum}\n"
        #        f"Current Plan: {self.plan}\n"
        #        f"Current Dataset code: {self.dataset_code}\n"
        #        f"Current Experiment code: {self.results_code}\n"
        #        f"Current Results: {self.exp_results}\n"
        #        f"Current Interpretation of results: {self.interpretation}\n"
        #    )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
            "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\n<Insert command here>\n``` where COMMAND is the specific command you want to run (e.g. REPORT, DIALOGUE).\n")

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "When you believe a good report has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```LATEX\nreport here\n```\n where report here is the actual report written in compilable latex to be transmitted and LATEX is just the word LATEX.\n"
            "Your report should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance. You must propagate this information accurately. You must also submit the report promptly. Do not delay too long.\n"
            "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        phase_str = (
            "You are directing a PhD student to help them write a report in latex based on results from an experiment, and you interact with them through dialogue.\n"
            "Your goal is to write a report in latex for an experiment. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the PhD student how they can write up the results and give their feedback to improve their thoughts.\n"
        )
        return phase_str

    def role_description(self):
        return "a computer science professor at a top university."


class PostdocAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["plan formulation", "results interpretation"]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "plan formulation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}",
            )
        elif phase == "results interpretation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}"
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "plan formulation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you believe a good plan has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```PLAN\nplan here\n```\n where plan here is the actual plan to be transmitted and PLAN is just the word PLAN. Plan here should provide a clear outline for how to achieve the task, including what machine learning models to use and implement, what types of datasets should be searched for and used to train the model, and the exact details of the experiment.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "Make sure not to produce too much dialogue and to submit an plan in reasonable time."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. PLAN, DIALOGUE).\n"
            )
        elif phase == "results interpretation":
            return (
                "When you believe a good interpretation has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```INTERPRETATION\ninterpretation here\n```\n where interpretation here is the actual interpretation to be transmitted and INTERPRETATION is just the word INTERPRETATION. Please provide an INTERPRETATION in a reasonable amount of time.\n"
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. INTERPRETATION, DIALOGUE).\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "plan formulation":
            phase_str = (
                "You are directing a PhD student to help them come up with a good plan, and you interact with them through dialogue.\n"
                "Your goal is to produce plans that would make good experiments for the given topic. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic. Your plans should provide a clear outline for how to achieve the task, including what machine learning models to use and implement, what types of datasets should be searched for and used to train the model, and the exact details of the experiment.\n"
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are directing a PhD student to help them come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the PhD student how they can interpret the results and give their feedback to improve their thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance. You must propagate this information accurately. You must also complete this in a reasonable amount of time and then submit your results.\n"
            )
        return phase_str

    def role_description(self):
        return "a computer science postdoctoral student at a top university."


class MLEngineerAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "data preparation",
            "running experiments",
        ]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\nPlan: {self.plan}",
                f"Current Plan: {self.plan}")
        #elif phase == "running experiments":
        #    return (
        #        sr_str,
        #        f"Current Literature Review: {self.lit_review_sum}\n"
        #        f"Current Plan: {self.plan}\n"
        #        f"Current Dataset code: {self.dataset_code}\n"
        #    )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            return (
                "You can produce code using the following command: ```python\ncode here\n```\n where code here is the actual code you will execute in a Python terminal, and python is just the word python. Try to incorporate some print functions. Do not use any classes or functions. If your code returns any errors, they will be provided to you, and you are also able to see print statements. You will receive all print statement results from the code. Make sure function variables are created inside the function or passed as a function parameter.\n"  # Try to avoid creating functions. 
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send, and DIALOGUE is just the word DIALOGUE.\n"
                "You also have access to HuggingFace datasets. You can search the datasets repository using the following command: ```SEARCH_HF\nsearch query here\n``` where search query here is the query used to search HuggingFace datasets, and SEARCH_HF is the word SEARCH_HF. This will return a list of HuggingFace dataset descriptions which can be loaded into Python using the datasets library. Your code MUST use an external HuggingFace directory.\n"
                "You MUST use a HuggingFace dataset in your code. DO NOT CREATE A MAIN FUNCTION. Try to make the code very simple.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. python, DIALOGUE, SEARCH_HF).\n")
        return ()

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            phase_str = (
                "You are a machine learning engineer being directed by a PhD student who will help you write the code, and you can interact with them through dialogue.\n"
                "Your goal is to produce code that prepares the data for the provided experiment. You should aim for simple code to prepare the data, not complex code. You should integrate the provided literature review and the plan and come up with code to prepare data for this experiment.\n"
            )
        return phase_str

    def role_description(self):
        return "a machine learning engineer working at a top university."



class SWEngineerAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "data preparation",
        ]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\nPlan: {self.plan}",
                f"Current Plan: {self.plan}")
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you and the ML engineer have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions. The submitted code must have a HuggingFace dataset import and must use an external HuggingFace dataset. If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. Submit the code early. You should submit the code ASAP.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        elif phase == "data preparation":
            phase_str = (
                "You are a software engineer directing a machine learning engineer, where the machine learning engineer will be writing the code, and you can interact with them through dialogue.\n"
                "Your goal is to help the ML engineer produce code that prepares the data for the provided experiment. You should aim for very simple code to prepare the data, not complex code. You should integrate the provided literature review and the plan and come up with code to prepare data for this experiment.\n"
            )
        return phase_str

    def role_description(self):
        return "a software engineer working at a top university."


class PhDStudentAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "literature review",
            "plan formulation",
            "running experiments",
            "results interpretation",
            "report writing",
            "report refinement",
        ]
        self.lit_review = []

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "plan formulation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}",)
        elif phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}"
            )
        elif phase == "results interpretation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}"
            )
        elif phase == "report refinement":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}\n"
                f"Current Interpretation of results: {self.interpretation}"
            )
        elif phase == "literature review":
            return sr_str
        else:
            return ""

    def requirements_txt(self):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a requirements.txt for a github repository for all of the code."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the requirements.txt below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "literature review":
            return (
                "To collect paper summaries, use the following command: ```SUMMARY\nSEARCH QUERY\n```\n where SEARCH QUERY is a string that will be used to find papers with semantically similar content and SUMMARY is just the word SUMMARY. Make sure your search queries are very short.\n"
                "To get the full paper text for an arXiv paper, use the following command: ```FULL_TEXT\narXiv paper ID\n```\n where arXiv paper ID is the ID of the arXiv paper (which can be found by using the SUMMARY command), and FULL_TEXT is just the word FULL_TEXT. Make sure to read the full text using the FULL_TEXT command before adding it to your list of relevant papers.\n"
                "If you believe a paper is relevant to the research project proposal, you can add it to the official review after reading using the following command: ```ADD_PAPER\narXiv_paper_ID\nPAPER_SUMMARY\n```\nwhere arXiv_paper_ID is the ID of the arXiv paper, PAPER_SUMMARY is a brief summary of the paper, and ADD_PAPER is just the word ADD_PAPER. You can only add one paper at a time. \n"
                "Make sure to use ADD_PAPER when you see a relevant paper. DO NOT use SUMMARY too many times."
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "Make sure to extensively discuss the experimental results in your summary.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. ADD_PAPER, FULL_TEXT, SUMMARY). Do not use the word COMMAND make sure to use the actual command, e.g. your command should look exactly like this: ```ADD_PAPER\ntext\n``` (where the command could be from ADD_PAPER, FULL_TEXT, SUMMARY)\n")
        elif phase == "plan formulation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        elif phase == "data preparation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you and the ML engineer have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions. The submitted code must have a HuggingFace dataset import and must use an external HuggingFace dataset. If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. Submit the code early. You should submit the code ASAP.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
        elif phase == "results interpretation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        #elif phase == "report writing":
        #    return (
        #        "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
        #        "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n")
        elif phase == "report refinement":
            return ""
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "literature review":
            phase_str = (
                "Your goal is to perform a literature review for the presented task and add papers to the literature review.\n"
                "You have access to arXiv and can perform two search operations: (1) finding many different paper summaries from a search query and (2) getting a single full paper text for an arXiv paper.\n"
            )
            rev_papers = "Papers in your review so far: " + " ".join([_paper["arxiv_id"] for _paper in self.lit_review])
            phase_str += rev_papers if len(self.lit_review) > 0 else ""
        elif phase == "plan formulation":
            phase_str = (
                "You are a PhD student being directed by a postdoc who will help you come up with a good plan, and you interact with them through dialogue.\n"
                "Your goal is to produce plans that would make good experiments for the given topic. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic. Your plans should provide a clear outline for how to achieve the task, including what machine learning models to use and implement, what types of datasets should be searched for and used to train the model, and the exact details of the experiment.\n"
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are a PhD student being directed by a postdoc who will help you come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the postdoc your interpretation and use their feedback to improve your thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance. You must propagate this information accurately.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
            )
        #elif phase == "report writing":
        #    phase_str = (
        #        "You are a PhD student being directed by a professor who will help you write a report based on results from an experiment, and you interact with them through dialogue.\n"
        #        "Your goal is to write a report for an experiment entirely in latex. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the professor how you can write up the results and receive their feedback to improve your thoughts.\n"
        #        "Your report should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance  in latex. You must propagate this information accurately.\n"
        #        "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
        #    )
        elif phase == "report refinement":
            phase_str = (
                "You are a PhD student who has submitted their paper to an ML conference called ICLR. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the conference.\n"
            )
        else:
            phase_str = ""
        return phase_str

    def role_description(self):
        return "a computer science PhD student at a top university."

    def add_review(self, review, arx_eng):
        try:
            arxiv_id, review_text = review.strip().split("\n", 1)
            full_text = arx_eng.retrieve_full_paper_text(arxiv_id)
            review_entry = {
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "summary": review_text,
            }
            self.lit_review.append(review_entry)
            return f"Successfully added paper {arxiv_id}", full_text
        except Exception as e:
            return f"Error trying to add review -- bad formatting, try again: {str(e)}", ""

    def format_review(self):
        return "Provided here is a literature review on this topic:\n" + "\n".join(
            f"arXiv ID: {_l['arxiv_id']}, Summary: {_l['summary']}"
            for _l in self.lit_review)



