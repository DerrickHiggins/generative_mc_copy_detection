from math import log2
from typing import Union
import operator
import numpy as np
import pandas as pd

## TODO
## - Recast as class
##   - Constructor takes dataframe
##   - Member data: alpha, results, answer_probs
## - Deal with global state
##   - Columns for responses (NUM_ITEMS)
##   - Column for student ID
##   - Column for Student Name

df = pd.read_csv("quiz-CS585MidtermS2020-full.csv").set_index("StudentID")
df = df.fillna("NA")


NUM_ITEMS = 42
DEFAULT_ALPHA = 0.5


answer_probs = []
for i in range(NUM_ITEMS):
    answer_probs.append(df[f"Stu{i+1}"].value_counts(normalize=True).to_dict())



def log_prob_ratio(s1: pd.Series, s2: pd.Series, alpha: float) -> float:
    """Calculate the log of the ratio of P(response_1, response2) under 2 models
    Model 1: Independence -- P_1(r1, r2) = P(r1) * P(r2)
    Model 2: Cheating -- P_2(r1, r2) = P(r1) * [ ALPHA * OneHot(r1)(r2) + (1-ALPHA) * P(r2)]

    param s1: pd.Series for student 1 response vector
    param s2: pd.Series for student 2 response vector 
    param alpha: probability of copying on a single item under copying hypothesis
    return: float representing log probability ratio of copying vs. independence hypothesis
    """
    p_i = 0
    p_c = 0
    for i in range(NUM_ITEMS):
        r1 = s1[f"Stu{i+1}"]
        r2 = s2[f"Stu{i+1}"]
        m1 = s1[f"Mark{i+1}"]
        m2 = s2[f"Mark{i+1}"]
        # Skip correct answers
        if m1 != "C" or m2 != "C":
            r1p = answer_probs[i][r1]
            r2p = answer_probs[i][r2]
            p_i += log2(r1p)
            p_i += log2(r2p)
            
            ans_match = float(r1 == r2)
            c1 = log2(r1p) + log2(alpha * ans_match + (1-alpha) * r2p)
            c2 = log2(r2p) + log2(alpha * ans_match + (1-alpha) * r1p)
            p_c += (c1 + c2) / 2
    return p_c - p_i



def get_score(df: pd.DataFrame, id1: Union[int, str], id2: Union[int, str], 
              alpha: float) -> float:
    s1 = df.loc[id1, :]
    s2 = df.loc[id2, :]
    return log_prob_ratio(s1, s2, alpha=alpha)



def get_copying_logprobs(df: pd.DataFrame, alpha: float=DEFAULT_ALPHA,
    student_sort_order: str="id") -> pd.DataFrame:
    """Perform analysis of student exam responses represented as a DataFrame
    to determine likelihood of copying answers.  The number reported for a
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
    results = {}
    for sid1 in df.index:
        results[sid1] = {sid2: get_score(df, sid1, sid2, alpha) if sid1 != sid2 else np.nan 
        for sid2 in df.index}

    if student_sort_order == "max_logprob":
        max_sims = {k: np.nanmax(list(v.values())) for k, v in results.items()}
        sid_order = sorted(list(df.index), key=lambda x: max_sims[x], reverse=True)
        results = {k:{kk: vv if sid_order.index(k) > sid_order.index(kk) 
                   else np.nan for kk, vv in v.items()} for k, v in results.items()}
    elif student_sort_order == "id":
        sid_order = sorted(list(df.index), reverse=False)
        results = {k:{kk: vv if k < kk else np.nan for kk, vv in v.items()} 
                   for k, v in results.items()}
    else:
        raise Exception("Acceptable values for `student_sort_order` are `id` or `max_logprob`")

    score_df = pd.DataFrame(results, columns=sid_order)
    score_df = score_df.sort_index(ascending=True, key=lambda x: [sid_order.index(y) for y in x])

    return score_df



# ## Find top pairs

score_tuples = [(s1, s2, score) for s1, vals in results.items() for s2, score in vals.items() if not np.isnan(score)]
score_tuples = sorted(score_tuples, key=operator.itemgetter(2), reverse=True)



name_dict = {}
for row in df.iterrows():
    name_dict[row[0]] = row[1].LastName


def show_top_scores(score_tuples, n=20):
    for s1, s2, score in score_tuples[:n]:
        print(f"{score:0.5f} {name_dict[s1]:20s} {name_dict[s2]:20s}")



