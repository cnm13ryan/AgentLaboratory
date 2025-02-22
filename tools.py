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
    Class for finding relevant Hugging Face datasets by searching dataset descriptions
    and scoring them based on textual similarity, likes, and downloads.
    """
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Initialize the dataset search with thresholds.
        :param like_thr: Minimum number of likes required.
        :param dwn_thr: Minimum number of downloads required.
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

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

        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return

        self.ds = self.ds.select(filtered_indices)
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        """
        Normalize an array to the range [0, 1] using min-max normalization.
        """
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieve the top N datasets that best match the query.
        The final score is a weighted sum of cosine similarity, normalized likes, and normalized downloads.
        """
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()
        cosine_similarities_norm = self._normalize(cosine_similarities)
        final_scores = (
            sim_w * cosine_similarities_norm +
            like_w * self.likes_norm +
            dwn_w * self.downloads_norm
        )
        top_indices = final_scores.argsort()[-N:][::-1]
        top_indices = [int(i) for i in top_indices]
        top_datasets = [self.ds[i] for i in top_indices]

        # Retrieve test/train split information via helper method.
        test_info, train_info = self._collect_ds_split_info(top_indices)
        for i, ds_item in enumerate(top_datasets):
            ds_item["has_test_set"] = test_info[top_indices[i]]["has"]
            ds_item["test_download_size"] = test_info[top_indices[i]]["download_size"]
            ds_item["test_element_size"] = test_info[top_indices[i]]["element_size"]
            ds_item["has_train_set"] = train_info[top_indices[i]]["has"]
            ds_item["train_download_size"] = train_info[top_indices[i]]["download_size"]
            ds_item["train_element_size"] = train_info[top_indices[i]]["element_size"]
        return top_datasets

    def _collect_ds_split_info(self, indices):
        """
        Helper function to collect test and train split information for datasets.
        :param indices: List of dataset indices to process.
        :return: Two dictionaries: test_info and train_info, keyed by dataset index.
        """
        test_info = {}
        train_info = {}
        for i in indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception:
                test_info[i] = {"has": False, "download_size": None, "element_size": None}
                train_info[i] = {"has": False, "download_size": None, "element_size": None}
                continue

            if dbuilder.splits is None:
                test_info[i] = {"has": False, "download_size": None, "element_size": None}
                train_info[i] = {"has": False, "download_size": None, "element_size": None}
                continue

            if "test" in dbuilder.splits:
                test_info[i] = {
                    "has": True,
                    "download_size": bytes2human(dbuilder.splits["test"].num_bytes),
                    "element_size": dbuilder.splits["test"].num_examples
                }
            else:
                test_info[i] = {"has": False, "download_size": None, "element_size": None}

            if "train" in dbuilder.splits:
                train_info[i] = {
                    "has": True,
                    "download_size": bytes2human(dbuilder.splits["train"].num_bytes),
                    "element_size": dbuilder.splits["train"].num_examples
                }
            else:
                train_info[i] = {"has": False, "download_size": None, "element_size": None}
        return test_info, train_info


class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        """
        Search Semantic Scholar for papers matching the query.
        Returns a list of formatted paper summaries.
        """
        paper_sums = []
        results = self.sch_engine.search_paper(query, limit=N, min_citation_count=3, open_access_pdf=True)
        for _i in range(len(results)):
            paper_sum = f"Title: {results[_i].title}\n"
            paper_sum += f"Abstract: {results[_i].abstract}\n"
            paper_sum += f"Citations: {results[_i].citationCount}\n"
            paper_sum += (f"Release Date: year {results[_i].publicationDate.year}, "
                          f"month {results[_i].publicationDate.month}, "
                          f"day {results[_i].publicationDate.day}\n")
            paper_sum += f"Venue: {results[_i].venue}\n"
            paper_sum += f"Paper ID: {results[_i].externalIds['DOI']}\n"
            paper_sums.append(paper_sum)
        return paper_sums


class ArxivSearch:
    def __init__(self):
        import arxiv
        self.arxiv_client = arxiv.Client()

    def _process_query(self, query: str) -> str:
        """
        Truncate the query to ensure it does not exceed the maximum length.
        """
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
        Search arXiv for papers matching the query.
        Returns a concatenated string of paper summaries.
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
        Download the PDF for a given arXiv paper and extract its text.
        """
        import arxiv
        pdf_text = ""
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
        paper.download_pdf(filename="downloaded-paper.pdf")
        reader = PdfReader("downloaded-paper.pdf")
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception as e:
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"
            pdf_text += f"--- Page {page_number} ---\n{text}\n"
        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)
        return pdf_text


def execute_code(code_str, timeout=60, MAX_LEN=1000):
    """
    Execute Python code in a restricted environment with a timeout and capture the output.
    """
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this."
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
        return (f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. "
                "You must reduce the time complexity of your code.")
    finally:
        sys.stdout = sys.__stdout__
    return output_capture.getvalue()[:MAX_LEN]
