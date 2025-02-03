from agents import *
from copy import copy
from common_imports import *
from mlesolver import MLESolver

import argparse
import pickle

DEFAULT_LLM_BACKBONE = "o1-mini"


class LaboratoryWorkflow:
    def __init__(
        self,
        research_topic,
        openai_api_key,
        max_steps=100,
        num_papers_lit_review=5,
        agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}",
        notes=None,
        human_in_loop_flag=None,
        compile_pdf=True,
        mlesolver_max_steps=3,
        papersolver_max_steps=5
    ):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """
        if notes is None:
            notes = []

        # --- Basic fields / user arguments ---
        self.research_topic = research_topic
        self.openai_api_key = openai_api_key
        self.max_steps = max_steps
        self.num_papers_lit_review = num_papers_lit_review
        self.model_backbone = agent_model_backbone
        self.notes = notes
        self.human_in_loop_flag = human_in_loop_flag
        self.compile_pdf = compile_pdf
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        # --- Additional workflow flags ---
        self.print_cost = True
        self.verbose = True
        self.save = True

        # --- Review overrides / counters ---
        self.review_override = True  # should review be overridden?
        self.review_total_steps = 0  # total steps to forcibly override
        self.review_ovrd_steps = 0   # how many steps so far

        # --- Timers / references ---
        self.arxiv_paper_exp_time = 3
        self.reference_papers = []

        # --- Phase definitions ---
        self.phases = [
            ("literature review",     ["literature review"]),
            ("plan formulation",      ["plan formulation"]),
            ("experimentation",       ["data preparation", "running experiments"]),
            ("results interpretation",["results interpretation", "report writing", "report refinement"]),
        ]
        self.phase_status = {}
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        # --- Phase model overrides ---
        self.phase_models = {}
        if isinstance(agent_model_backbone, str):
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif isinstance(agent_model_backbone, dict):
            self.phase_models = agent_model_backbone

        # --- Simple stats tracking ---
        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0},
            "plan formulation":       {"time": 0.0, "steps": 0.0},
            "data preparation":       {"time": 0.0, "steps": 0.0},
            "running experiments":    {"time": 0.0, "steps": 0.0},
            "results interpretation": {"time": 0.0, "steps": 0.0},
            "report writing":         {"time": 0.0, "steps": 0.0},
            "report refinement":      {"time": 0.0, "steps": 0.0},
        }

        # --- Instantiate Agents ---
        self.reviewers = ReviewersAgent(
            model=self.model_backbone,
            notes=self.notes,
            openai_api_key=self.openai_api_key
        )
        self.phd = PhDStudentAgent(
            model=self.model_backbone,
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key
        )
        self.postdoc = PostdocAgent(
            model=self.model_backbone,
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key
        )
        self.professor = ProfessorAgent(
            model=self.model_backbone,
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key
        )
        self.ml_engineer = MLEngineerAgent(
            model=self.model_backbone,
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key
        )
        self.sw_engineer = SWEngineerAgent(
            model=self.model_backbone,
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key
        )

        # --- Remove / Create directories ---
        remove_figures()
        remove_directory("research_dir")
        if not os.path.exists("state_saves"):
            os.mkdir(os.path.join(".", "state_saves"))
        os.mkdir(os.path.join(".", "research_dir"))
        os.mkdir(os.path.join("./research_dir", "src"))
        os.mkdir(os.path.join("./research_dir", "tex"))

    def perform_research(self):
        """
        Loop through all research phases/subtasks and perform them in order.
        @return: None
        """
        if not self.openai_api_key:
            # Guard Clause
            print("No API key provided; aborting research workflow.")
            return

        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase

            if self.verbose:
                print(f"{'*' * 50}\nBeginning phase: {phase}\n{'*' * 50}")

            for subtask in subtasks:
                if self.verbose:
                    print(f"{'&' * 30}\nBeginning subtask: {subtask}\n{'&' * 30}")

                # Set subtask-specific model if overridden in self.phase_models
                if isinstance(self.phase_models, dict):
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else:
                        self.set_model(f"{DEFAULT_LLM_BACKBONE}")

                # Dispatch to the subtask method if not already done
                if not self.phase_status.get(subtask, False):
                    repeat = False

                    if subtask == "literature review":
                        repeat = True
                        while repeat:
                            repeat = self.literature_review()
                        self.phase_status[subtask] = True

                    elif subtask == "plan formulation":
                        repeat = True
                        while repeat:
                            repeat = self.plan_formulation()
                        self.phase_status[subtask] = True

                    elif subtask == "data preparation":
                        repeat = True
                        while repeat:
                            repeat = self.data_preparation()
                        self.phase_status[subtask] = True

                    elif subtask == "running experiments":
                        repeat = True
                        while repeat:
                            repeat = self.running_experiments()
                        self.phase_status[subtask] = True

                    elif subtask == "results interpretation":
                        repeat = True
                        while repeat:
                            repeat = self.results_interpretation()
                        self.phase_status[subtask] = True

                    elif subtask == "report writing":
                        repeat = True
                        while repeat:
                            repeat = self.report_writing()
                        self.phase_status[subtask] = True

                    elif subtask == "report refinement":
                        return_to_exp_phase = self.report_refinement()
                        if not return_to_exp_phase:
                            # done, save & return
                            if self.save:
                                self.save_state(subtask)
                            return

                        # If we must go back to earlier phases:
                        self.set_agent_attr("second_round", return_to_exp_phase)
                        self.set_agent_attr("prev_report", copy(self.phd.report))
                        self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                        self.set_agent_attr("prev_results_code", copy(self.phd.results_code))
                        self.set_agent_attr("prev_interpretation", copy(self.phd.interpretation))

                        # Reset statuses so we can re-run them
                        self.phase_status["plan formulation"] = False
                        self.phase_status["data preparation"] = False
                        self.phase_status["running experiments"] = False
                        self.phase_status["results interpretation"] = False
                        self.phase_status["report writing"] = False
                        self.phase_status["report refinement"] = False

                        self.perform_research()

                    # Save state after subtask
                    if self.save:
                        self.save_state(subtask)

                    phase_end_time = time.time()
                    phase_duration = phase_end_time - phase_start_time
                    print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                    self.statistics_per_phase[subtask]["time"] = phase_duration

    def literature_review(self):
        """
        Perform literature review phase.
        Possibly repeated until the user (or agent) is satisfied.
        @return: (bool) whether the subtask should be repeated
        """
        arx_eng = ArxivSearch()
        max_tries = self.max_steps * 5  # lit review often requires extra steps

        # Initial response from PhD
        resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.8)
        if self.verbose:
            print(resp, "\n~~~~~~~~~~~")

        # Iterate until we have enough papers or run out of tries
        for _i in range(max_tries):
            feedback = ""

            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                feedback = f"You requested arXiv papers related to {query}:\n{papers}"

            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                # expiration marker
                arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n"
                arxiv_paper += arx_eng.retrieve_full_paper_text(query) + "```"
                feedback = arxiv_paper

            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                feedback, text = self.phd.add_review(query, arx_eng)
                # also track references if needed
                if len(self.reference_papers) < self.num_papers_lit_review:
                    self.reference_papers.append(text)

            # Check if completed
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # Summarize final lit review
                lit_review_sum = self.phd.format_review()

                # If human in loop, ask for acceptance
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    if retry:
                        self.phd.lit_review = []
                        return retry

                if self.verbose:
                    print(self.phd.lit_review_sum)

                self.set_agent_attr("lit_review_sum", lit_review_sum)
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

            # Next iteration
            resp = self.phd.inference(
                self.research_topic, "literature review",
                feedback=feedback, step=_i + 1, temp=0.8
            )
            if self.verbose:
                print(resp, "\n~~~~~~~~~~~")

        raise Exception("Max tries during phase: Literature Review")

    def plan_formulation(self):
        """
        Perform plan formulation phase.
        Possibly repeated until user/agent is satisfied.
        @return: (bool) whether the subtask should be repeated
        """
        max_tries = self.max_steps
        dialogue = ""

        for _i in range(max_tries):
            # Postdoc
            resp = self.postdoc.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose:
                print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = ""

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#" * 40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry:
                        return retry
                self.set_agent_attr("plan", plan)
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            # PhD
            resp = self.phd.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose:
                print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "PhD Dialogue:", dialogue, "#" * 40, "\n")

        raise Exception("Max tries during phase: Plan Formulation")

    def data_preparation(self):
        """
        Perform data preparation phase.
        Possibly repeated until user/agent is satisfied.
        @return: (bool) whether the subtask should be repeated
        """
        max_tries = self.max_steps
        ml_feedback, ml_dialogue, swe_feedback, ml_command = "", "", "", ""
        hf_engine = HFDataSearch()

        for _i in range(max_tries):
            # SW Engineer
            swe_input = f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}"
            if ml_feedback:
                swe_input += f"\nFeedback provided to the ML agent: {ml_feedback}"

            resp = self.sw_engineer.inference(
                self.research_topic, "data preparation",
                feedback=swe_input, step=_i
            )
            swe_feedback = ""
            swe_dialogue = ""

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose:
                    print("#" * 40, f"\nSW Engineer Dialogue: {dialogue}", "\n", "#" * 40)

            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code, timeout=60)
                if self.verbose:
                    print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error! Must fix.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry:
                            return retry

                    save_to_file("./research_dir/src", "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            # ML Engineer
            ml_input = f"{swe_dialogue}\n"
            if ml_feedback:
                ml_input += f"Feedback from previous command: {ml_feedback}\n"

            resp = self.ml_engineer.inference(self.research_topic, "data preparation", feedback=ml_input, step=_i)
            ml_feedback, ml_dialogue, ml_command = "", "", ""

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose:
                    print("#" * 40, f"\nML Engineer Dialogue: {dialogue}", "#" * 40, "\n")

            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code, timeout=120)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose:
                    print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")

            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(hf_engine.results_str(hf_engine.retrieve_ds(hf_query)))
                ml_command = f"HF search command: {hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"

        raise Exception("Max tries during phase: Data Preparation")

    def running_experiments(self):
        """
        Perform the experiments (training, inference, etc.).
        Possibly repeated until user/agent is satisfied.
        @return: (bool) whether the subtask should be repeated
        """
        # experiment notes
        experiment_notes = [
            _note["note"] for _note in self.ml_engineer.notes
            if "running experiments" in _note["phases"]
        ]
        experiment_notes_str = (
            f"Notes for the task objective: {experiment_notes}\n" 
            if len(experiment_notes) > 0 else ""
        )

        # Initialize MLESolver
        solver = MLESolver(
            dataset_code=self.ml_engineer.dataset_code,
            notes=experiment_notes_str,
            insights=self.ml_engineer.lit_review_sum,
            max_steps=self.mlesolver_max_steps,
            plan=self.ml_engineer.plan,
            openai_api_key=self.openai_api_key,
            llm_str=self.phase_models["running experiments"]
        )

        # run initialization
        solver.initial_solve()
        for _ in range(self.mlesolver_max_steps - 1):
            solver.solve()

        # best code
        code = "\n".join(solver.best_codes[0][0])
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]

        # Execute code
        execute_code(code)

        if self.verbose:
            print(f"Running experiments completed, reward function score: {score}")

        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("running experiments", code)
            if retry:
                return retry

        # Save final code
        save_to_file("./research_dir/src", "run_experiments.py", code)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Interpret the experimental results.
        Possibly repeated until user/agent is satisfied.
        @return: (bool) whether the subtask should be repeated
        """
        max_tries = self.max_steps
        dialogue = ""

        for _i in range(max_tries):
            # Postdoc
            resp = self.postdoc.inference(
                self.research_topic, "results interpretation", feedback=dialogue, step=_i
            )
            if self.verbose:
                print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = ""

            if "```DIALOGUE" in resp:
                postdoc_dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoc: {postdoc_dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#" * 40)

            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry:
                        return retry
                self.set_agent_attr("interpretation", interpretation)
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False

            # PhD
            resp = self.phd.inference(
                self.research_topic, "results interpretation", feedback=dialogue, step=_i
            )
            if self.verbose:
                print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            if "```DIALOGUE" in resp:
                phd_dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {phd_dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "PhD Dialogue:", dialogue, "#" * 40, "\n")

        raise Exception("Max tries during phase: Results Interpretation")

    def report_writing(self):
        """
        Prepare the research report.
        Possibly repeated until user/agent is satisfied.
        @return: (bool) whether the subtask should be repeated
        """
        report_notes = [
            _note["note"] for _note in self.ml_engineer.notes
            if "report writing" in _note["phases"]
        ]
        report_notes_str = (
            f"Notes for the task objective: {report_notes}\n" 
            if len(report_notes) > 0 else ""
        )

        # Import PaperSolver here if needed
        from papersolver import PaperSolver

        # Initialize the PaperSolver
        solver = PaperSolver(
            notes=report_notes_str,
            max_steps=self.papersolver_max_steps,
            plan=self.phd.plan,
            exp_code=self.phd.results_code,
            exp_results=self.phd.exp_results,
            insights=self.phd.interpretation,
            lit_review=self.phd.lit_review,
            ref_papers=self.reference_papers,
            topic=self.research_topic,
            openai_api_key=self.openai_api_key,
            llm_str=self.phase_models["report writing"],
            compile_pdf=self.compile_pdf
        )

        solver.initial_solve()
        for _ in range(self.papersolver_max_steps):
            solver.solve()

        # best report
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]

        if self.verbose:
            print(f"Report writing completed, reward function score: {score}")

        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry:
                return retry

        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file("./research_dir", "readme.md", readme)
        save_to_file("./research_dir", "report.txt", report)
        self.reset_agents()
        return False

    def report_refinement(self):
        """
        Refine the final report by collecting feedback from reviewers.
        @return: (bool) whether to repeat the entire pipeline
        """
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)

        if self.human_in_loop_flag["report refinement"]:
            print(
                f"Reviews from a set of three reviewers: {reviews}\n"
                "Would you like to be done or go back and improve experiments?\n"
            )
            user_input = input("(y) for go back, (n) for complete: ").strip().lower()
            if user_input == "y":
                self.set_agent_attr("reviewer_response", str(reviews))
                return True
            return False
        else:
            # Automated approach
            review_prompt = (
                f"Reviews: {reviews}\n"
                "Type 'y' to go back and improve experiments, 'n' to finalize."
            )
            if self.review_override:
                # artificially step each time until we decide to finalize
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic,
                    phase="report refinement",
                    feedback=review_prompt,
                    step=0
                )
                response = response.lower().strip()[0] if len(response) > 0 else ""

            if response == "n":
                if self.verbose:
                    print("*" * 40, "\n", "REVIEW COMPLETE", "\n", "*" * 40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Reviews: {reviews}")
                return True

            raise Exception("Model did not respond with y/n in report_refinement()")

    # ----------------------------------------------------------
    #                      HELPER METHODS
    # ----------------------------------------------------------

    def set_model(self, model):
        """ Set agent models (except ReviewersAgent which is separate). """
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def set_agent_attr(self, attr, obj):
        """
        Set attribute for all relevant agents.
        """
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def save_state(self, phase):
        """
        Save the entire LaboratoryWorkflow object state for the given phase.
        """
        phase = phase.replace(" ", "_")
        with open(f"state_saves/{phase}.pkl", "wb") as f:
            pickle.dump(self, f)

    def reset_agents(self):
        """ Reset internal state of all agents. """
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def human_in_loop(self, phase, phase_prod):
        """
        Prompt the human for acceptance or further feedback on a subtask result.
        @return: (bool) whether to repeat the subtask
        """
        print(f"\nResult of phase [{phase}]:\n{phase_prod}")
        while True:
            answer = input("\nAre you satisfied? (Y or N): ").strip().lower()
            if answer == "y":
                return False
            elif answer == "n":
                notes_for_agent = input("Please provide notes to improve the result: ")
                self.reset_agents()
                self.notes.append({"phases": [phase], "note": notes_for_agent})
                return True
            else:
                print("Invalid response. Please type 'Y' or 'N'.")


########################################################
#                   ARG PARSING & MAIN                 #
########################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")

    parser.add_argument(
        '--copilot-mode',
        type=str,
        default="False",
        help='Enable human interaction mode.'
    )
    parser.add_argument(
        '--deepseek-api-key',
        type=str,
        help='Provide the DeepSeek API key.'
    )
    parser.add_argument(
        '--load-existing',
        type=str,
        default="False",
        help='Load existing state if True; otherwise start a new workflow.'
    )
    parser.add_argument(
        '--load-existing-path',
        type=str,
        help='Path to load existing state, e.g. state_saves/results_interpretation.pkl'
    )
    parser.add_argument(
        '--research-topic',
        type=str,
        help='Specify the research topic.'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Provide the OpenAI API key.'
    )
    parser.add_argument(
        '--compile-latex',
        type=str,
        default="True",
        help='Compile latex into pdf during paper writing phase.'
    )
    parser.add_argument(
        '--llm-backend',
        type=str,
        default="o1-mini",
        help='Backend LLM to use for agents in Agent Laboratory.'
    )
    parser.add_argument(
        '--language',
        type=str,
        default="English",
        help='Language to operate in.'
    )
    parser.add_argument(
        '--num-papers-lit-review',
        type=str,
        default="5",
        help='Number of papers to summarize in the literature review stage.'
    )
    parser.add_argument(
        '--mlesolver-max-steps',
        type=str,
        default="3",
        help='Total number of MLESolver steps.'
    )
    parser.add_argument(
        '--papersolver-max-steps',
        type=str,
        default="5",
        help='Total number of PaperSolver steps.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    llm_backend = args.llm_backend
    human_mode = args.copilot_mode.lower() == "true"
    compile_pdf = args.compile_latex.lower() == "true"
    load_existing = args.load_existing.lower() == "true"

    try:
        num_papers_lit_review = int(args.num_papers_lit_review.lower())
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        papersolver_max_steps = int(args.papersolver_max_steps.lower())
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")
    try:
        mlesolver_max_steps = int(args.mlesolver_max_steps.lower())
    except Exception:
        raise Exception("args.mlesolver_max_steps must be a valid integer!")

    api_key = os.getenv('OPENAI_API_KEY') or args.api_key
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key

    if args.api_key is not None and os.getenv('OPENAI_API_KEY') is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.deepseek_api_key is not None and os.getenv('DEEPSEEK_API_KEY') is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key:
        raise ValueError(
            "API key must be provided via --api-key / --deepseek-api-key or "
            "the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable."
        )

    # If no research topic is given, ask the user
    if human_mode or args.research_topic is None:
        research_topic = input("Please name an experiment idea for AgentLaboratory to perform: ")
    else:
        research_topic = args.research_topic

    task_notes_LLM = [
        {"phases": ["plan formulation"],
         "note": f"You should come up with a plan for TWO experiments."},
        {"phases": ["plan formulation", "data preparation", "running experiments"],
         "note": "Please use gpt-4o-mini for your experiments."},
        {"phases": ["running experiments"],
         "note": f'Use the following code to inference gpt-4o-mini:\n'
                 f'from openai import OpenAI\n'
                 f'os.environ["OPENAI_API_KEY"] = "{api_key}"\n'
                 f'client = OpenAI()\n'
                 f'completion = client.chat.completions.create(\n'
                 f'  model="gpt-4o-mini-2024-07-18", messages=messages)\n'
                 f'answer = completion.choices[0].message.content\n'},
        {"phases": ["running experiments"],
         "note": f"You have access to only gpt-4o-mini using the OpenAI API, do not use openai.ChatCompletion.create."},
        {"phases": ["running experiments"],
         "note": "Use a small dataset of ~100 data points to keep run-time small."},
        {"phases": ["data preparation", "running experiments"],
         "note": "You are running on a MacBook laptop. You can use 'mps' with PyTorch."},
        {"phases": ["data preparation", "running experiments"],
         "note": "Generate figures with colorful, artistic design."},
    ]
    task_notes_LLM.append(
        {"phases": [
            "literature review", "plan formulation", "data preparation",
            "running experiments", "results interpretation",
            "report writing", "report refinement"
         ],
         "note": f"Write everything in {args.language}."}
    )

    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "report writing":         llm_backend,
        "results interpretation": llm_backend,
        "report refinement":      llm_backend,  # match the subtask name
    }

    if load_existing:
        load_path = args.load_existing_path
        if load_path is None:
            raise ValueError("Please provide path to load existing state.")
        with open(load_path, "rb") as f:
            lab = pickle.load(f)
    else:
        lab = LaboratoryWorkflow(
            research_topic=research_topic,
            notes=task_notes_LLM,
            agent_model_backbone=agent_models,
            human_in_loop_flag=human_in_loop,
            openai_api_key=api_key,
            compile_pdf=compile_pdf,
            num_papers_lit_review=num_papers_lit_review,
            papersolver_max_steps=papersolver_max_steps,
            mlesolver_max_steps=mlesolver_max_steps,
        )

    lab.perform_research()
