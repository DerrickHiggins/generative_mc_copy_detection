# A generative model for detection of copying for in-class multiple-choice exams

A tool to identify potential instances of copying between students administered
a multiple-choice exam.  Unlike other copying detection methods (such as
[CopyDetect](https://cran.r-project.org/web/packages/CopyDetect/)), this model does
not use item-response theory (IRT), so it can still be applied even when the
number of students is not sufficient to calibrate an IRT model.

## Installation

```
# Clone the repo
git clone https://github.com/DerrickHiggins/generative_mc_copy_detection.git

# Install package
cd generative_mc_copy_detection && pip install .
```

## Data Preparation

The package assumes that student response data for the test is provided in the
default export format of the [ZipGrade](https://www.zipgrade.com/) platform.  This is a CSV format with one row per student, including

* a student ID column
* a student name column
* `N` response columns with each student's answers to the `N` items
    * For example, "A", "B", "C", "D", or "ACD" (in the case of multiple-selection items)
* `N` mark columns with each student's score for the `N` items
    * "C" to indicate correct responses, "X" for incorrect responses, or "P" for partial credit

## Usage

To perform an analysis on the dataset to assess which pairs of student test answer sets have the highest likelihood of resulting from copying, initialize the `GenMCCopyDetector` object.  Methods such as `print_top_scores` can be used to access the metrics.

```python
from genmccd import GenMCCopyDetector
cd = GenMCCopyDetector()
cd.print_top_scores(n=10)
```

Example output might look like the following:

```
19.87418 Adrianna Alsbrooks   Isela Ioli          
13.34704 Britany Buskirk      Delmar Dupree       
8.35350 Britany Buskirk      Marcelina Modrak    
8.35350 Delmar Dupree        Marcelina Modrak    
7.13334 Vada Velverton       Adrianna Alsbrooks  
7.13334 Vada Velverton       Isela Ioli          
6.62258 Magdalene Mossor     Zachery Zeitler     
6.12024 Kayleigh Kalland     Sanford Stogner     
5.85412 Suzy Spaniel         Ute Unga            
3.68424 Coral Catchings      Fredricka Fioravanti
```
The metric reported for each student pair is <img src="https://render.githubusercontent.com/render/math?math=\log_2(\mathrm{P}[A_1,A_2%20|%20\mathrm{copying}])%20-%20\log_2(\mathrm{P}[A_1,A_2%20|%20\mathrm{independent}])">.  So a metric of 6.1 means that the students' test answers are <img src="https://render.githubusercontent.com/render/math?math=2^{6.1}"> times more likely under the copying scenario than they would be under the independent work scenario.

See the [examples](examples/) folder for some sample datasets and results.

## Generative Model

This copy-detection method is based on an underlying generative model governing the distribution of pairs of student answer sets for a test.  We consider all answer pairs <img src="https://render.githubusercontent.com/render/math?math=A_{S_i},%20A_{S_j}"> for student pairs <img src="https://render.githubusercontent.com/render/math?math=S_i,%20S_j"> to be generated independently of one another, according to one of two processes.  

Under one process, the students' answers are generated independently of one another, with students answering randomly to each question according to the observed distribution of answer choices for each question.  (This corresponds to students working independently and _not_ copying.)  Under the other process, _one_ student generates answers according to the observed distribution of answer choices for each question, while the other copies the first student's response &alpha;% of the time and generates an independent answer (1 - &alpha;)% of the time.  (This corresponds to copying behavior.)

Formally, **XX**

### Assumptions and Limitations

- Alpha parameter
- Assumption that student answers agree only by chance or because of copying
   - Item difficulty, distractors -- and student ability
   - Similar study patterns
- Derivation of response probabilities from observed data