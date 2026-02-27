from evaluation.eval_vicuna import (
    run_eval as run_eval_vicuna,
    reorg_answer_file as reorg_answer_file_vicuna
)
from evaluation.eval_llama3 import (
    run_eval as run_eval_llama3,
    reorg_answer_file as reorg_answer_file_llama3,
)
from evaluation.eval_tulu import (
    run_eval as run_eval_tulu,
    reorg_answer_file as reorg_answer_file_tulu,
)

run_evals = {
    "vicuna": run_eval_vicuna,
    "llama3": run_eval_llama3,
    "tulu": run_eval_tulu,
}
reorg_answer_files = {
    "vicuna": reorg_answer_file_vicuna,
    "llama3": reorg_answer_file_llama3,
    "tulu": reorg_answer_file_tulu,
}