from utils import *

import time
import os
import io
import sys
import traceback
import numpy as np
import concurrent.futures
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt

from pypdf import PdfReader
from datasets import load_dataset, load_dataset_builder
from psutil._common import bytes2human
from semanticscholar import SemanticScholar
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


class HFDataSearch:
    """
    Class for finding relevant Hugging Face datasets by searching descriptions,
    and weighting by likes/downloads.
    """
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        :param like_thr: Minimum threshold for dataset likes
        :param dwn_thr: Minimum threshold for dataset downloads
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

        # Iterate over the dataset and filter based on criteria
        for idx, item in enumerate(self.ds):
            likes = int(item['likes']) if item['likes'] is not None else 0
            downloads = int(item['downloads']) if item['downloads'] is not None else 0

            if likes >= self.like_thr and downloads >= self.dwn_thr:
                description = item['description']
                if isinstance(description, str) and description.strip():
                    filtered_indices.append(idx)
                    filtered_descriptions.append(description)
                    filtered_likes.append(likes)
                    filtered_downloads.append(downloads)

        # Check if any datasets meet all criteria
        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return  # Exit the constructor

        # Filter the datasets using the collected indices
        self.ds = self.ds.select(filtered_indices)

        # Update descriptions, likes, and downloads
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)

        # Normalize likes and downloads
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)

        # Vectorize the descriptions
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        """
        Simple min-max normalization. Returns an array in [0, 1].
        """
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query,
        weighted by similarity, likes, and downloads.
        """
        # 1. Early exit if no datasets
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        # 2. Calculate similarity scores
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()
        # Normalize the similarity scores for fair weighting
        cosine_similarities_norm = self._normalize(cosine_similarities)

        # 3. Combine scores
        final_scores = (
            sim_w * cosine_similarities_norm +
            like_w * self.likes_norm +
            dwn_w * self.downloads_norm
        )

        # 4. Sort and retrieve the top N
        top_indices = final_scores.argsort()[-N:][::-1]
        top_indices = [int(i) for i in top_indices]  # Ensure Python int
        top_datasets = [self.ds[i] for i in top_indices]

        # 5. Gather test & train set info
        test_info, train_info = self._collect_ds_split_info(top_indices)

        # 6. Annotate final results
        for i, ds_idx in enumerate(top_indices):
            top_datasets[i]["has_test_set"] = test_info[ds_idx]["has"]
            top_datasets[i]["test_download_size"] = test_info[ds_idx]["download_size"]
            top_datasets[i]["test_element_size"] = test_info[ds_idx]["element_size"]
            top_datasets[i]["has_train_set"] = train_info[ds_idx]["has"]
            top_datasets[i]["train_download_size"] = train_info[ds_idx]["download_size"]
            top_datasets[i]["train_element_size"] = train_info[ds_idx]["element_size"]

        return top_datasets

    def _collect_ds_split_info(self, indices):
        """
        For each dataset index in `indices`, attempt to retrieve
        size/availability info for test and train splits.
        Returns two dictionaries (test_info, train_info), keyed by ds index.
        """
        test_info = {}
        train_info = {}

        for i in indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception:
                # If it fails, mark everything as not available
                test_info[i] = {"has": False, "download_size": None, "element_size": None}
                train_info[i] = {"has": False, "download_size": None, "element_size": None}
                continue

            # If splits is None, also mark as not available
            if dbuilder.splits is None:
                test_info[i] = {"has": False, "download_size": None, "element_size": None}
                train_info[i] = {"has": False, "download_size": None, "element_size": None}
                continue

            # Check test
            if "test" in dbuilder.splits:
                test_info[i] = {
                    "has": True,
                    "download_size": bytes2human(dbuilder.splits["test"].num_bytes),
                    "element_size": dbuilder.splits["test"].num_examples
                }
            else:
                test_info[i] = {"has": False, "download_size": None, "element_size": None}

            # Check train
            if "train" in dbuilder.splits:
                train_info[i] = {
                    "has": True,
                    "download_size": bytes2human(dbuilder.splits["train"].num_bytes),
                    "element_size": dbuilder.splits["train"].num_examples
                }
            else:
                train_info[i] = {"has": False, "download_size": None, "element_size": None}

        return test_info, train_info

    def results_str(self, results):
        """
        Provide results as list of human-readable strings.
        """
        result_strs = []
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs


class SemanticScholarSearch:
    """
    Class for searching papers via Semantic Scholar.
    """
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        """
        Search for papers by query string. Returns a list of formatted strings.
        """
        paper_sums = []
        results = self.sch_engine.search_paper(query, limit=N, min_citation_count=3, open_access_pdf=True)
        for paper in results:
            paper_sum = f"Title: {paper.title}\n"
            paper_sum += f"Abstract: {paper.abstract}\n"
            paper_sum += f"Citations: {paper.citationCount}\n"
            if paper.publicationDate:
                paper_sum += (f"Release Date: year {paper.publicationDate.year}, "
                              f"month {paper.publicationDate.month}, "
                              f"day {paper.publicationDate.day}\n")
            else:
                paper_sum += "Release Date: Unknown\n"
            paper_sum += f"Venue: {paper.venue}\n"
            if "DOI" in paper.externalIds:
                paper_sum += f"Paper ID: {paper.externalIds['DOI']}\n"
            paper_sums.append(paper_sum)
        return paper_sums


class ArxivSearch:
    """
    Class for searching papers via arXiv.
    """
    def __init__(self):
        # Construct the default API client.
        import arxiv
        self.arxiv_client = arxiv.Client()

    def _process_query(self, query: str) -> str:
        """Truncate query to fit within MAX_QUERY_LENGTH."""
        MAX_QUERY_LENGTH = 300
        if len(query) <= MAX_QUERY_LENGTH:
            return query

        words = query.split()
        processed_query = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break

        return ' '.join(processed_query)

    def find_papers_by_str(self, query, N=20):
        """
        Search arXiv by query string, return up to N results.
        """
        import arxiv
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                paper_sums = []
                # `results` is a generator
                for r in self.arxiv_client.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)
                time.sleep(2.0)
                return "\n".join(paper_sums)
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)
                    continue

        return None

    def retrieve_full_paper_text(self, query):
        """
        Download the PDF from arXiv by ID and extract text content.
        """
        import arxiv
        pdf_text = ""
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
        # Download the PDF
        paper.download_pdf(filename="downloaded-paper.pdf")

        reader = PdfReader("downloaded-paper.pdf")
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception:
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"

            pdf_text += f"--- Page {page_number} ---\n{text}\n"

        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)
        return pdf_text


def execute_code(code_str, timeout=60, MAX_LEN=1000):
    """
    Execute Python code in a restricted environment,
    with a given timeout and maximum output length.
    """
    # Preliminary checks for blocked patterns
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took too long. Program terminated."
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed. Please remove it."

    output_capture = io.StringIO()
    sys.stdout = output_capture
    exec_globals = globals()

    def run_code():
        try:
            exec(code_str, exec_globals)
        except Exception as e:
            output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
            traceback.print_exc(file=output_capture)

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_code)
            future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        sys.stdout = sys.__stdout__
        return (f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout "
                f"limit of {timeout} seconds. You must reduce the time complexity.")
    finally:
        sys.stdout = sys.__stdout__

    return output_capture.getvalue()[:MAX_LEN]
