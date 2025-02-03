from utils import *
from tools import *
from inference import *
import re
import json

# ----------------------------------------------------------------------
# Helper to build a "reviewer system prompt" (used in get_score).
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
        """
        + template_instructions
    )
    base_str = (
        "You are an AI researcher who is reviewing a paper that was submitted "
        "to a prestigious ML venue. Be critical and cautious."
    )
    # reviewer_type is appended to clarify the style or tone.
    return f"{base_str} {reviewer_type}\n{neurips_form}"

# ----------------------------------------------------------------------
# JSON extraction helper
def extract_json_between_markers(llm_output):
    """
    Attempt to extract valid JSON from the string between ```json ... ``` markers.
    If not found, fallback to any JSON-like content and attempt to parse.
    """
    # -------------------------------------------------------------------------
    # Guard clause for empty output:
    if not llm_output:
        return None
    
    # -------------------------------------------------------------------------
    # Try to find JSON content between ```json ... ``` blocks.
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern_fallback = r"\{.*?\}"
        matches = re.findall(json_pattern_fallback, llm_output, re.DOTALL)

    # -------------------------------------------------------------------------
    # Parse each match carefully
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
                # Continue if still invalid
                continue

    # -------------------------------------------------------------------------
    # No valid JSON found
    return None

# ----------------------------------------------------------------------
# Score function that uses a reward model LLM
def get_score(
    outlined_plan, 
    latex, 
    reward_model_llm, 
    reviewer_type=None, 
    attempts=3, 
    openai_api_key=None
):
    """
    Given a plan (outlined_plan) and associated LaTeX (latex),
    use a reward model LLM (reward_model_llm) to generate a review,
    parse it into JSON, and derive a performance score.
    """
    # -------------------------------------------------------------------------
    # Guard clauses for any critical missing input:
    if not outlined_plan:
        return 0.0, "Missing 'outlined_plan'.", False
    
    if not latex:
        return 0.0, "Missing 'latex'.", False
    
    if not reward_model_llm:
        return 0.0, "Missing 'reward_model_llm' (LLM model name).", False
    
    # -------------------------------------------------------------------------
    e = ""
    for _attempt in range(attempts):
        try:
            if reviewer_type is None:
                reviewer_type = ""

            # Build the system prompt using our new helper
            sys = _build_reviewer_system_prompt(reviewer_type)

            # -----------------------------------------------------------------
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

            # -----------------------------------------------------------------
            # Parse JSON from the LLM output
            review_json = extract_json_between_markers(scoring)
            if not review_json:
                return 0.0, "Could not parse valid JSON review.", False

            # -----------------------------------------------------------------
            # Convert string-based fields into numeric for scoring
            overall = int(review_json["Overall"]) / 10
            soundness = int(review_json["Soundness"]) / 4
            confidence = int(review_json["Confidence"]) / 5
            contribution = int(review_json["Contribution"]) / 4
            presentation = int(review_json["Presentation"]) / 4
            clarity = int(review_json["Clarity"]) / 4
            originality = int(review_json["Originality"]) / 4
            quality = int(review_json["Quality"]) / 4
            significance = int(review_json["Significance"]) / 4

            # -----------------------------------------------------------------
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

            # -----------------------------------------------------------------
            # Max possible sum of the weights
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

            # -----------------------------------------------------------------
            # Weighted average scaled to 10
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

            # -----------------------------------------------------------------
            # Return performance & entire text for debugging/inspection
            return performance, f"The performance of your submission is: {performance}\n{scoring}", True

        except Exception as ex:
            e = str(ex)
            # Try next attempt if any remain
            continue

    # -------------------------------------------------------------------------
    # If all attempts failed, return last known exception
    return 0.0, e, False

# ----------------------------------------------------------------------
# BaseAgent with extracted helper methods for building system/user prompts
class BaseAgent:
    """
    A foundational agent with shared properties/methods for specialized roles.
    """

    def __init__(
        self, 
        model="gpt-4o-mini", 
        notes=None, 
        max_steps=100, 
        openai_api_key=None
    ):
        if model is None:
            raise ValueError("model must be specified (string).")
        if max_steps <= 0:
            raise ValueError("max_steps should be > 0.")

        self.notes = notes if notes else []
        self.max_steps = max_steps
        self.model = model
        self.openai_api_key = openai_api_key

        # Internal states
        self.phases = []
        self.plan = ""
        self.report = ""
        self.history = []
        self.prev_comm = ""
        self.prev_report = ""
        self.exp_results = ""
        self.dataset_code = ""
        self.results_code = ""
        self.lit_review_sum = ""
        self.interpretation = ""
        self.prev_exp_results = ""
        self.reviewer_response = ""
        self.prev_results_code = ""
        self.prev_interpretation = ""

        # Control flags / config
        self.second_round = False
        self.max_hist_len = 15

    def set_model_backbone(self, model):
        if not model:
            raise ValueError("model cannot be empty.")
        self.model = model

    @staticmethod
    def clean_text(text):
        """
        Fix minor formatting issues in the text output.
        """
        if not text:
            return ""
        return text.replace("```\n", "```")

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
        # Guard clauses:
        if not research_topic:
            return "No research topic provided."
        if not phase:
            return "No phase provided."
        if step < 0:
            return "Step cannot be negative."

        # Create context for the given phase
        context_str = self.context(phase)

        # Convert history to a single string
        history_str = "\n".join([h[1] for h in self.history])

        # Possibly instruct to finish if near the end
        complete_str = ""
        if step / (self.max_steps - 1) > 0.7:
            complete_str = "You must finish this task and submit as soon as possible!"

        # Possibly parse special feedback
        # (Example for expiration steps, if present in feedback)
        # Not strictly needed unless you want to unify that logic
        # for every agent. We'll keep it here for demonstration.
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

    def inference(
        self, 
        research_topic, 
        phase, 
        step, 
        feedback="", 
        temp=None
    ):
        """
        Main inference loop for the agent, generating a system prompt
        (via build_system_prompt) and a user prompt (via build_user_prompt),
        then querying the underlying LLM with query_model().
        """
        # Build system prompt
        sys_prompt = self.build_system_prompt(phase)
        # Build user prompt
        user_prompt = self.build_user_prompt(research_topic, phase, step, feedback)

        # Query the LLM
        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=user_prompt,
            temp=temp,
            openai_api_key=self.openai_api_key
        )

        # Clean and store response
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp

        # Keep history (including a possible ephemeral expiration)
        steps_exp = None
        if feedback and "```EXPIRATION" in feedback:
            try:
                steps_exp_line = feedback.split("\n")[0]
                steps_exp = int(steps_exp_line.replace("```EXPIRATION ", ""))
            except ValueError:
                steps_exp = None

        self.history.append(
            (steps_exp, f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {model_resp}")
        )

        # Decrement expiration counters & remove expired
        for i in reversed(range(len(self.history))):
            if self.history[i][0] is not None:
                new_exp = self.history[i][0] - 1
                self.history[i] = (new_exp, self.history[i][1])
                if new_exp < 0:
                    self.history.pop(i)

        # Limit max history length
        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)

        return model_resp

    def reset(self):
        """
        Clear stored conversation/history.
        """
        self.history.clear()
        self.prev_comm = ""

    # -------------------------------------------------------------------------
    # Abstract methods to be implemented by subclasses:

    def context(self, phase):
        raise NotImplementedError("Subclasses should implement 'context(phase)'.")

    def phase_prompt(self, phase):
        raise NotImplementedError("Subclasses should implement 'phase_prompt(phase)'.")

    def role_description(self):
        raise NotImplementedError("Subclasses should implement 'role_description()'.")

    def command_descriptions(self, phase):
        raise NotImplementedError("Subclasses should implement 'command_descriptions(phase)'.")

    def example_command(self, phase):
        raise NotImplementedError("Subclasses should implement 'example_command(phase)'.")


# ----------------------------------------------------------------------
# Specialized Agents
class PhDStudentAgent(BaseAgent):
    """
    A specialized agent that acts as a PhD student, focusing on:
    - Literature review
    - Plan formulation
    - Running experiments
    - Results interpretation
    - Report writing/refinement
    """

    def __init__(
        self, 
        model="gpt4omini", 
        notes=None, 
        max_steps=100, 
        openai_api_key=None
    ):
        super().__init__(model, notes, max_steps, openai_api_key)

        # PhD phases
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
        sr_str = ""
        if self.second_round:
            sr_str = (
                "The following are results from the previous experiments:\n"
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )

        if phase == "plan formulation":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}"
            )
        elif phase == "data preparation":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}\n"
                + f"Current Plan: {self.plan}"
            )
        elif phase == "results interpretation":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}\n"
                + f"Current Plan: {self.plan}\n"
                + f"Current Dataset code: {self.dataset_code}\n"
                + f"Current Experiment code: {self.results_code}\n"
                + f"Current Results: {self.exp_results}"
            )
        elif phase == "report refinement":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}\n"
                + f"Current Plan: {self.plan}\n"
                + f"Current Dataset code: {self.dataset_code}\n"
                + f"Current Experiment code: {self.results_code}\n"
                + f"Current Results: {self.exp_results}\n"
                + f"Current Interpretation of results: {self.interpretation}"
            )
        elif phase == "literature review":
            return sr_str
        else:
            return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "literature review":
            phase_str = (
                "Your goal is to perform a literature review for the presented task "
                "and add papers to the literature review. You have access to arXiv. "
                "You can search short queries, fetch full text, and decide whether to add the paper."
            )
            rev_papers = "Papers in your review so far: " + " ".join(
                [p["arxiv_id"] for p in self.lit_review]
            )
            if self.lit_review:
                phase_str += "\n" + rev_papers

        elif phase == "plan formulation":
            phase_str = (
                "You are a PhD student being directed by a postdoc to create a research plan. "
                "Propose a simple but effective experimental plan integrating the literature."
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are a PhD student interpreting experiment results, possibly with help from a postdoc. "
                "Use the experiment code, results, and literature to form a coherent interpretation."
            )
        elif phase == "report refinement":
            phase_str = (
                "You are refining your report for a submission to an ML conference. "
                "Incorporate feedback, prior results, and produce a polished final version."
            )
        elif phase == "report writing":
            phase_str = (
                "You are writing a report in LaTeX based on the experiments run. "
                "Include key numbers, metrics, and details. Interact with your professor for guidance."
            )
        elif phase == "running experiments":
            phase_str = (
                "You are orchestrating or checking the code to run experiments. "
                "Collaborate with the ML engineer or relevant persons to gather results."
            )
        else:
            phase_str = ""

        return phase_str

    def role_description(self):
        return "a computer science PhD student at a top university."

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "literature review":
            return (
                "Use these commands to fetch and add papers:\n"
                "```SUMMARY\nSEARCH QUERY\n``` to find relevant papers.\n"
                "```FULL_TEXT\narXiv paper ID\n``` to get an entire paper.\n"
                "```ADD_PAPER\narXiv_paper_ID\nPAPER_SUMMARY\n``` to add a paper to the review.\n"
                "You can only use one command per inference turn."
            )
        elif phase == "plan formulation":
            return (
                "You can produce dialogue with: ```DIALOGUE\nsome text\n```\n"
                "Only one command per inference turn."
            )
        elif phase == "data preparation":
            return (
                "Use: ```DIALOGUE\nsome text\n``` to discuss code with the ML engineer.\n"
                "When finalizing, use: ```SUBMIT_CODE\nactual code\n```\n"
                "One command per turn, only. The code must be simple and rely on HF dataset."
            )
        elif phase == "results interpretation":
            return (
                "Use: ```DIALOGUE\nsome text\n``` to talk with a postdoc.\n"
                "One command per turn."
            )
        elif phase == "report refinement":
            return (
                "In this phase, you refine the report. No special commands except dialogue or final submission if your code environment supports it."
            )
        elif phase == "report writing":
            return (
                "You can produce dialogue to discuss the writing with your professor. "
                "In some frameworks, you might finalize the LaTeX with a command like ```LATEX\nfull report\n```."
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def add_review(self, review, arx_eng):
        """
        Helper method to add a new paper to this agent's in-memory lit_review.
        """
        if not review:
            return "No review text provided.", ""

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
            return f"Error trying to add review -- bad formatting. {str(e)}", ""

    def format_review(self):
        """
        Return a multi-paper summary for the current literature review.
        """
        if not self.lit_review:
            return "No papers added yet."

        review_str = "Provided here is a literature review on this topic:\n"
        for paper in self.lit_review:
            review_str += (
                f"arXiv ID: {paper['arxiv_id']}, "
                f"Summary: {paper['summary']}\n"
            )
        return review_str

    def requirements_txt(self):
        """
        Generate a requirements.txt given the conversation history and knowledge.
        """
        sys_prompt = (
            f"You are {self.role_description()}.\n"
            "Task: Integrate all knowledge, code, and notes to produce a requirements.txt."
        )
        history_str = "\n".join([h[1] for h in self.history])
        prompt = (
            f"History: {history_str}\n{'~' * 10}\n"
            f"Please produce the requirements.txt below:\n"
        )
        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt,
            openai_api_key=self.openai_api_key
        )
        return model_resp


class PostdocAgent(BaseAgent):
    """
    A specialized agent playing the role of a postdoctoral researcher:
    - Guides plan formulation
    - Helps interpret experimental results
    """

    def __init__(
        self, 
        model="gpt4omini", 
        notes=None, 
        max_steps=100, 
        openai_api_key=None
    ):
        super().__init__(model, notes, max_steps, openai_api_key)

        self.phases = [
            "plan formulation",
            "results interpretation"
        ]

    def context(self, phase):
        sr_str = ""
        if self.second_round:
            sr_str = (
                "The following are results from the previous experiments:\n"
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )

        if phase == "plan formulation":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}"
            )
        elif phase == "results interpretation":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}\n"
                + f"Current Plan: {self.plan}\n"
                + f"Current Dataset code: {self.dataset_code}\n"
                + f"Current Experiment code: {self.results_code}\n"
                + f"Current Results: {self.exp_results}"
            )
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "plan formulation":
            return (
                "You are a postdoc helping formulate a plan with the PhD student. "
                "Focus on a simple, well-defined experiment that references the existing literature."
            )
        elif phase == "results interpretation":
            return (
                "You are a postdoc guiding the PhD student in interpreting results. "
                "Look at the code, metrics, and tie them back to the original research plan/literature."
            )
        return ""

    def role_description(self):
        return "a computer science postdoctoral researcher at a top university."

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "plan formulation":
            return (
                "You can produce dialogue with: ```DIALOGUE\nsome text\n```\n"
                "When you believe a good plan is formed, use ```PLAN\nplan details\n```\n"
                "One command per turn."
            )
        elif phase == "results interpretation":
            return (
                "You can produce dialogue with: ```DIALOGUE\nsome text\n```\n"
                "When a good interpretation is formed, use ```INTERPRETATION\nexplanation\n```\n"
                "One command per turn."
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()


class MLEngineerAgent(BaseAgent):
    """
    A specialized agent playing the role of a machine learning engineer:
    - Data preparation
    - Running experiments
    """

    def __init__(
        self,
        model="gpt4omini",
        notes=None,
        max_steps=100,
        openai_api_key=None
    ):
        super().__init__(model, notes, max_steps, openai_api_key)

        self.phases = [
            "data preparation",
            "running experiments",
        ]

    def context(self, phase):
        sr_str = ""
        if self.second_round:
            sr_str = (
                "The following are results from the previous experiments:\n"
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )

        if phase == "data preparation":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}\n"
                + f"Current Plan: {self.plan}"
            )
        elif phase == "running experiments":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}\n"
                + f"Current Plan: {self.plan}\n"
                + f"Current Dataset code: {self.dataset_code}\n"
            )
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "data preparation":
            return (
                "You are the ML engineer preparing data for the experiment. "
                "Integrate the plan and relevant literature. Keep code simple."
            )
        elif phase == "running experiments":
            return (
                "You are running experiments based on the plan. "
                "Implement the ML pipeline, training, etc."
            )
        return ""

    def role_description(self):
        return "a machine learning engineer working at a top university."

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "data preparation":
            return (
                "Use: ```python\ncode\n``` for code. Must include HF dataset usage.\n"
                "Use: ```DIALOGUE\nsome text\n``` for short discussion.\n"
                "Use: ```SEARCH_HF\nsearch query\n``` to search HuggingFace datasets.\n"
                "One command per turn."
            )
        elif phase == "running experiments":
            return (
                "Similar approach, but code for running experiments might be more complex. "
                "Use: ```python\ncode\n``` to produce code, or ```DIALOGUE\ntext\n``` to discuss."
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()


class SWEngineerAgent(BaseAgent):
    """
    A specialized agent playing the role of a software engineer:
    - Overlaps with data preparation, but focuses on code handoff
    """

    def __init__(
        self,
        model="gpt4omini",
        notes=None,
        max_steps=100,
        openai_api_key=None
    ):
        super().__init__(model, notes, max_steps, openai_api_key)

        self.phases = [
            "data preparation",
        ]

    def context(self, phase):
        sr_str = ""
        if self.second_round:
            sr_str = (
                "The following are results from the previous experiments:\n"
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )

        if phase == "data preparation":
            return (
                sr_str
                + f"Current Literature Review: {self.lit_review_sum}\n"
                + f"Plan: {self.plan}"
            )
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "data preparation":
            return (
                "You are a software engineer guiding the ML engineer to produce code for data prep. "
                "Coordinate with them to finalize the code that uses a HF dataset."
            )
        return ""

    def role_description(self):
        return "a software engineer working at a top university."

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "data preparation":
            return (
                "Use: ```DIALOGUE\nsome text\n``` to coordinate. "
                "Finally, when ready, use: ```SUBMIT_CODE\ncode\n``` to finalize. "
                "One command per turn."
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()


class ProfessorAgent(BaseAgent):
    """
    A specialized agent playing the role of a professor:
    - Guides report writing with the PhD student
    """

    def __init__(
        self, 
        model="gpt4omini", 
        notes=None, 
        max_steps=100, 
        openai_api_key=None
    ):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["report writing"]

    def context(self, phase):
        # Example: minimal context for the professor
        if self.second_round:
            return (
                "The following are second-round results or feedback:\n"
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "report writing":
            return (
                "You are a professor guiding the PhD student in writing up the final experiment results in LaTeX."
            )
        return ""

    def role_description(self):
        return "a computer science professor at a top university."

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "report writing":
            return (
                "Use: ```DIALOGUE\nsome text\n``` to discuss the report. "
                "When a final version is ready, use: ```LATEX\nfull latex here\n``` to submit."
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "Example: ```DIALOGUE\nLet's add more detail to Section 2.\n```"
        )

    def generate_readme(self):
        """
        Generate a readme.md for a GitHub repository summarizing the entire project.
        """
        sys_prompt = (
            f"You are {self.role_description()}.\n"
            f"Here is the written paper:\n{self.report}\n"
            "Task: Summarize all knowledge, code, reports, and notes into a readme.md."
        )
        history_str = "\n".join([h[1] for h in self.history])
        prompt = (
            f"History: {history_str}\n{'~' * 10}\n"
            "Please produce the readme in markdown below:\n"
        )

        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt,
            openai_api_key=self.openai_api_key
        )
        return model_resp.replace("```markdown", "")


class ReviewersAgent:
    """
    An agent that manages multiple 'reviewers' (in terms of personality/perspective)
    and uses `get_score` to score a submission.
    """

    def __init__(
        self, 
        model="gpt-4o-mini", 
        notes=None, 
        openai_api_key=None
    ):
        if notes is None:
            self.notes = []
        else:
            self.notes = notes

        self.model = model
        self.openai_api_key = openai_api_key

    def inference(self, plan, report):
        """
        Invoke the `get_score` function with different reviewer personalities.
        """
        # Guard clauses:
        if not plan:
            return "No plan provided to ReviewersAgent."
        if not report:
            return "No report provided to ReviewersAgent."

        reviewer_1 = (
            "You are a harsh but fair reviewer and expect good experiments that "
            "lead to insights for the research topic."
        )
        review_1 = get_score(
            outlined_plan=plan,
            latex=report,
            reward_model_llm=self.model,
            reviewer_type=reviewer_1,
            openai_api_key=self.openai_api_key
        )

        reviewer_2 = (
            "You are a harsh and critical but fair reviewer who is looking for "
            "an idea that would be impactful in the field."
        )
        review_2 = get_score(
            outlined_plan=plan,
            latex=report,
            reward_model_llm=self.model,
            reviewer_type=reviewer_2,
            openai_api_key=self.openai_api_key
        )

        reviewer_3 = (
            "You are a harsh but fair open-minded reviewer that is looking for "
            "novel ideas that have not been proposed before."
        )
        review_3 = get_score(
            outlined_plan=plan,
            latex=report,
            reward_model_llm=self.model,
            reviewer_type=reviewer_3,
            openai_api_key=self.openai_api_key
        )

        return (
            f"Reviewer #1:\n{review_1},\n"
            f"Reviewer #2:\n{review_2},\n"
            f"Reviewer #3:\n{review_3}"
        )
