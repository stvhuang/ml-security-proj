from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.datasets import HuggingFaceDataset
from textattack.goal_functions import UntargetedClassification
from textattack.loggers import CSVLogger
from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.search_methods import GreedySearch, GreedyWordSwapWIR
from textattack.shared import Attack
from textattack.transformations import WordDeletion, WordSwapMaskedLM
from transformers import BertConfig, BertForSequenceClassification

from dataset import load_dataset

tokenizer = AutoTokenizer("hfl/chinese-macbert-base")
config = BertConfig.from_pretrained("hfl/chinese-macbert-base")
config.output_attentions = False
config.output_token_type_ids = False
model = BertForSequenceClassification.from_pretrained(
    "/data/model/hfl/chinese-macbert-base/chnsenticorp_0.998141",
    config=config,
)
dataset = load_dataset()
dataset = HuggingFaceDataset(dataset)
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# goal function
goal_function = UntargetedClassification(model_wrapper)

# constraints
constraints = [
    MaxWordsPerturbed(20),
]

# transformation
# transformation = WordSwapMaskedLM(method="bae", max_candidates=50, min_confidence=0.0)
transformation = WordDeletion()

# search methods
# search_method = GreedyWordSwapWIR(wir_method="delete")
search_method = GreedySearch()

# attack
attack = Attack(goal_function, constraints, transformation, search_method)
print(attack)

# https://textattack.readthedocs.io/en/latest/2notebook/1_Introduction_and_Transformations.html#Using-the-attack
results_iterable = attack.attack_dataset(dataset)
logger = CSVLogger(color_method=None)
num_successes = 0

while num_successes < 10:
    result = next(results_iterable)

    if isinstance(result, SuccessfulAttackResult):
        logger.log_attack_result(result)
        num_successes += 1
        print(f"{num_successes} of 10 successes complete.")

print(logger.df.loc[:, ["original_text", "perturbed_text"]])
