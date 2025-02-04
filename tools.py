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
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant Hugging Face datasets.
        :param like_thr: Minimum likes threshold.
        :param dwn_thr: Minimum downloads threshold.
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
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
        """Perform min-max normalization to scale array values to [0, 1]."""
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query,
        weighted by cosine similarity, likes, and downloads.
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

        # Retrieve split information (test/train) later via helper method (to be refactored in Commit 3)
        has_test_set = []
        has_train_set = []
        ds_size_info = []
        for i in top_indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            if dbuilder.splits is None:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            has_test, has_train = "test" in dbuilder.splits, "train" in dbuilder.splits
            has_test_set.append(has_test)
            has_train_set.append(has_train)
            test_dwn_size, test_elem_size = None, None
            train_dwn_size, train_elem_size = None, None
            if has_test:
                test_dwn_size = bytes2human(dbuilder.splits["test"].num_bytes)
                test_elem_size = dbuilder.splits["test"].num_examples
            if has_train:
                train_dwn_size = bytes2human(dbuilder.splits["train"].num_bytes)
                train_elem_size = dbuilder.splits["train"].num_examples
            ds_size_info.append((test_dwn_size, test_elem_size, train_dwn_size, train_elem_size))
        for i, ds_item in enumerate(top_datasets):
            ds_item["has_test_set"] = has_test_set[i]
            ds_item["has_train_set"] = has_train_set[i]
            ds_item["test_download_size"] = ds_size_info[i][0]
            ds_item["test_element_size"] = ds_size_info[i][1]
            ds_item["train_download_size"] = ds_size_info[i][2]
            ds_item["train_element_size"] = ds_size_info[i][3]
        return top_datasets

    def results_str(self, results):
        """
        Format the dataset search results into human-readable strings.
        """
        result_strs = []
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs


class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
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

    def retrieve_full_paper_text(self, query):
        pass


class ArxivSearch:
    def __init__(self):
        import arxiv
        self.arxiv_client = arxiv.Client()

    def _process_query(self, query: str) -> str:
        """Truncate the query to MAX_QUERY_LENGTH characters while preserving whole words."""
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
    Execute Python code in a restricted environment with a timeout.
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
