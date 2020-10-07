from math import log2
from typing import Union
import operator
import re
import numpy as np
import pandas as pd
import networkx as nx


DEFAULT_ALPHA = 0.8


class GenMCCopyDetector:

    input_df = None
    alpha = DEFAULT_ALPHA
    answer_probs = None
    pairwise_logprobs = None

    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float = DEFAULT_ALPHA,
        id_col: str = "StudentID",
        name_col: str = "LastName",
        response_cols: Union[list, str] = "infer",
        mark_cols: Union[list, str] = "infer",
    ):
        """Perform analysis of student exam responses represented as a DataFrame
        to determine likelihood of copying answers.

        param df: Student response DataFrame for the exam, in ZipGrade format
        param alpha: Probability of copying on a single item under copying hypothesis
        param id_col: Column with student id
        param name_col: Column with student name
        response_cols: Columns with student responses to each item
          (or "infer" if in ZipGrade format)
        mark_cols: Columns with mark indicators for response to each item
          (or "infer" if in ZipGrade format)
        """
        self.alpha = alpha
        self.id_col = id_col
        self.name_col = name_col

        if type(response_cols) == str and response_cols == "infer":
            self.response_cols = list(
                sorted([x for x in df.columns if re.match(r"Stu\d+$", x)])
            )
        else:
            self.response_cols = response_cols

        if type(mark_cols) == str and mark_cols == "infer":
            self.mark_cols = list(
                sorted([x for x in df.columns if re.match(r"Mark\d+$", x)])
            )
        else:
            self.mark_cols = mark_cols

        self.input_df = df.set_index(id_col)
        self.input_df = self.input_df.fillna("NA")

        # Initialize response probabilities for item options
        self.answer_probs = []
        for col in self.response_cols:
            self.answer_probs.append(
                self.input_df[col].value_counts(normalize=True).to_dict()
            )

        # Initialize pairwise logprobs
        self.pairwise_logprobs = {}
        for sid1 in self.input_df.index:
            self.pairwise_logprobs[sid1] = {
                sid2: self.get_score(sid1, sid2) if sid1 != sid2 else np.nan
                for sid2 in self.input_df.index
            }

    def log_prob_ratio(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate the log of the ratio of P(response_1, response2) under 2 models
        Model 1: Independence -- P_1(r1, r2) = P(r1) * P(r2)
        Model 2: Cheating -- P_2(r1, r2) = P(r1) * [ ALPHA * OneHot(r1)(r2) + (1-ALPHA) * P(r2)]

        param s1: pd.Series for student 1 response vector
        param s2: pd.Series for student 2 response vector
        return: float representing log probability ratio of copying vs. independence hypothesis
        """
        p_i = 0
        p_c = 0
        for i in range(len(self.response_cols)):
            rcol = self.response_cols[i]
            mcol = self.mark_cols[i]
            r1 = s1[rcol]
            r2 = s2[rcol]
            m1 = s1[mcol]
            m2 = s2[mcol]
            # Skip correct answers
            if m1 != "C" or m2 != "C":
                r1p = self.answer_probs[i][r1]
                r2p = self.answer_probs[i][r2]
                p_i += log2(r1p)
                p_i += log2(r2p)

                ans_match = float(r1 == r2)
                ## Here we take the average of logprob for s1 copying from s2 or vice-versa
                # c1 = log2(r1p) + log2(self.alpha * ans_match + (1 - self.alpha) * r2p)
                # c2 = log2(r2p) + log2(self.alpha * ans_match + (1 - self.alpha) * r1p)
                # p_c += (c1 + c2) / 2
                ## But we can simplify, since c1 is always equal to c2. Either
                ##  * r1 == r2 in which case r1p == r2p, or
                ##  * r1 != r2 in which case ans_match == 0 and 
                ##    c1 == c2 == log2((1 - self.alpha) * r1p * r2p)
                p_c += log2(r1p) + log2(self.alpha * ans_match + (1 - self.alpha) * r2p)
        return p_c - p_i

    def get_score(self, id1: Union[int, str], id2: Union[int, str]) -> float:
        s1 = self.input_df.loc[id1, :]
        s2 = self.input_df.loc[id2, :]
        return self.log_prob_ratio(s1, s2)

    def get_copying_logprobs(self, student_sort_order: str = "id") -> pd.DataFrame:
        """Return DataFrame with results of analysis. The number reported for a
        student pair S1 S2 is the log probabilty ratio of the likelihood of their
        response vector pair given that they were engaging in copying, to the
        likelihood of their response vector pair given that they were working
        independently.

        param df: Student response DataFrame for the exam, in ZipGrade format
        param alpha: probability of copying on a single item under copying hypothesis
        param student_sort_order: "id" or "max_logprob". Determines whether
        rows/columns in dataframe will be ordered by student ID or by the likelihood
        of copying
        return: DataFrame with logprobs for student pairs
        """
        if student_sort_order == "max_logprob":
            max_sims = {
                k: np.nanmax(list(v.values()))
                for k, v in self.pairwise_logprobs.items()
            }
            sid_order = sorted(
                list(self.input_df.index), key=lambda x: max_sims[x], reverse=True
            )
            results = {
                k: {
                    kk: vv if sid_order.index(k) < sid_order.index(kk) else np.nan
                    for kk, vv in v.items()
                }
                for k, v in self.pairwise_logprobs.items()
            }
        elif student_sort_order == "id":
            sid_order = sorted(list(self.input_df.index), reverse=False)
            results = {
                k: {kk: vv if k < kk else np.nan for kk, vv in v.items()}
                for k, v in self.pairwise_logprobs.items()
            }
        else:
            raise Exception(
                "Acceptable values for `student_sort_order` are `id` or `max_logprob`"
            )

        score_df = pd.DataFrame(results, columns=sid_order)
        score_df = score_df.sort_index(
            ascending=True, key=lambda x: [sid_order.index(y) for y in x]
        )

        return score_df

    def print_top_scores(self, n: int = 20):
        """Print students with top copying logprobs
        param n: Number of students to print
        """
        score_tuples = [
            (s1, s2, score)
            for s1, vals in self.pairwise_logprobs.items()
            for s2, score in vals.items()
            if not np.isnan(score) and s1 < s2
        ]
        score_tuples = sorted(score_tuples, key=operator.itemgetter(2), reverse=True)

        name_dict = {}
        for row in self.input_df.iterrows():
            name_dict[row[0]] = row[1][self.name_col]
        for s1, s2, score in score_tuples[:n]:
            print(f"{score:0.5f} {name_dict[s1]:20s} {name_dict[s2]:20s}")

    def get_graph(self, threshold: float = 0):
        """Return NetworkX graph with set of students whose logprob
        with another student is > threshold

        param threshold: minimum logprob for inclusion in set
        """
        g = nx.Graph()
        for sid1, v in self.pairwise_logprobs.items():
            for sid2, lp in v.items():
                if lp > threshold:
                    g.add_node(sid1)
                    g.add_node(sid2)
                    g.add_edge(sid1, sid2, weight=lp)
        return g
