import os

import hydra
from omegaconf import OmegaConf

from source.helper.LabelDescriptionHelper import LabelDescriptionHelper
from source.helper.PromptOptimizerHelper import PromptOptimizerHelper
from source.helper.RankingAggregationHelper import RankingAggregationHelper
from source.helper.RankingFusionHelper import RankingFusionHelper
from source.helper.SparseRetrieverHelper import SparseRetrieverHelper
from source.helper.retriever.RetrieverEvalHelper import RetrieverEvalHelper
from source.helper.retriever.RetrieverFitHelper import RetrieverFitHelper
from source.helper.retriever.RetrieverPredictHelper import RetrieverPredictHelper


def sparse_retrieve(params):
    SparseRetrieverHelper(params).run()


def fit(params):
    if params.model.type == "retriever":
        RetrieverFitHelper(params).run()


def predict(params):
    if params.model.type == "retriever":
        helper = RetrieverPredictHelper(params)
        helper.perform_predict()


def eval(params):
    if params.model.type == "retriever":
        helper = RetrieverEvalHelper(params)
        helper.perform_eval()


def aggregate(params):
    RankingAggregationHelper(params).run()


def fuse(params):
    RankingFusionHelper(params).run()


def prompt_opt(params):
    PromptOptimizerHelper(params).run()


def label_desc(params):
    LabelDescriptionHelper(params).run()


@hydra.main(config_path="setting", config_name="setting.yaml", version_base=None)
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "sparse_retrieve" in params.tasks:
        sparse_retrieve(params)

    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)

    if "fuse" in params.tasks:
        fuse(params)

    if "aggregate" in params.tasks:
        aggregate(params)

    if "prompt_opt" in params.tasks:
        prompt_opt(params)

    if "label_desc" in params.tasks:
        label_desc(params)


if __name__ == '__main__':
    perform_tasks()
