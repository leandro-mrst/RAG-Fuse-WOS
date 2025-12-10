import heapq
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, \
    TQDMProgressBar
from ranx import evaluate, Qrels, Run, fuse
from transformers import AutoTokenizer


class Helper:
    def __int__(self, params):
        self.params = params

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.params.model.tokenizer.architecture
        )

    def get_logger(self, fold_idx):
        return loggers.WandbLogger(
            project=self.params.log.project,
            save_dir=self.params.log.dir,
            name=f"{self.params.model.name}_{self.params.data.name}_{fold_idx}_exp"
        )

    # def get_logger(self, fold_idx):
    #     return loggers.TensorBoardLogger(
    #         save_dir=self.params.log.dir,
    #         name=f"{self.params.model.name}_{self.params.data.name}_{fold_idx}_exp"
    #     )

    def get_progress_bar_callback(self):
        return TQDMProgressBar(
            refresh_rate=self.params.trainer.progress_bar_refresh_rate,
            process_position=0
        )

    def get_lr_monitor(self):
        return LearningRateMonitor(logging_interval='step')

    def get_early_stopping_callback(self):
        return EarlyStopping(
            monitor='val_MRR',
            patience=self.params.trainer.patience,
            min_delta=self.params.trainer.min_delta,
            mode='max'
        )

    def get_model_checkpoint_callback(self, fold_idx):
        return ModelCheckpoint(
            monitor="val_MRR",
            dirpath=self.params.model_checkpoint.dir,
            filename=f"{self.params.model.name}_{self.params.data.name}_{fold_idx}",
            save_top_k=1,
            save_weights_only=True,
            mode="max"
        )

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"] = 1.0
            relevance_map[f"text_{text_idx}"] = d
        return relevance_map

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
        return metrics

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as cls_file:
            return pickle.load(cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as cls_file:
            return pickle.load(cls_file)

    def _load_samples(self):
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            return pickle.load(samples_file)

    def _load_split_samples(self, fold_idx, split):
        split_ids = self._load_split_ids(fold_idx, split)
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            samples = pickle.load(samples_file)
            return [sample for sample in samples if sample["idx"] in split_ids]

    def _get_ids(self, fold_idx, split):
        with open(f"{self.params.data.dir}fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def _load_split_ids(self, fold_idx, split):
        with open(f"{self.params.data.dir}fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
            return set(pickle.load(ids_file))

    def _min_max_normalize(self, labels_scores):
        values = list(labels_scores.values())
        max_value = max(values)
        min_value = min(values)
        return {key: (value - min_value) / (max_value - min_value) for key, value in labels_scores.items()}

    # def _normalize_ranking(self, ranking):
    #     for text_idx, labels_scores in ranking.items():
    #         ranking[text_idx] = self._min_max_normalize(labels_scores)
    #     return ranking

    # def _merge_rankings(self, ranking_1, ranking_2, fold_idx):
    #     merged_ranking = {
    #         fold_idx: {}
    #     }
    #
    #     for cls in ["tail", "head"]:
    #         merged_ranking[fold_idx][cls] = {}
    #         r1 = self._normalize_ranking(ranking_1[fold_idx][cls])
    #         r2 = self._normalize_ranking(ranking_2[fold_idx]["test"][cls])
    #
    #         for text_idx in set(r1).union(r2):
    #             if text_idx in r1 and text_idx in r2:
    #                 merged_ranking[fold_idx][cls][text_idx] = {x: r1[text_idx].get(x, 0) + r2[text_idx].get(x, 0)
    #                                                            for x
    #                                                            in
    #                                                            set(r1[text_idx]).union(r2[text_idx])}
    #             elif text_idx in r1:
    #                 merged_ranking[fold_idx][cls][text_idx] = r1[text_idx]
    #
    #             elif text_idx in r2:
    #                 merged_ranking[fold_idx][cls][text_idx] = r2[text_idx]
    #             else:
    #                 raise Exception("text_idx must be in r1 or r2.")
    #
    #     return merged_ranking

    # def _load_ranking(self, fold_idx):
    #     logging.info(f"Loading {self.params.retriever.sparse.name} ranking.")
    #     with open(f"{self.params.ranking.dir}{self.params.retriever.sparse.name}.rnk", "rb") as sparse_ranking_file:
    #         sparse_ranking = pickle.load(sparse_ranking_file)
    #
    #     logging.info(f"Loading {self.params.retriever.dense.name} ranking.")
    #     with open(f"{self.params.ranking.dir}{self.params.retriever.dense.name}.rnk", "rb") as dense_ranking_file:
    #         dense_ranking = pickle.load(dense_ranking_file)
    #
    #     fused_ranking = self._fuse_rankings(dense_ranking, sparse_ranking, fold_idx)
    #
    #     sliced_ranking = self._slice_ranking(fused_ranking, fold_idx, num_labels=64)
    #
    #     with open(self.params.ranking.dir + "Hybrid" + "_" + self.params.data.name + ".rnk", "wb") as rankings_file:
    #         pickle.dump(sliced_ranking, rankings_file)
    #
    #     r = self._eval_ranking(sliced_ranking, fold_idx)
    #     logging.info(f"Effectiveness of (sparse + dense) ranking:\n{r}.")
    #     return sliced_ranking

    def _slice_ranking(self, ranking, fold_idx, num_labels):
        sliced_ranking = {
            fold_idx: {}
        }
        for split in ["train", "val", "test"]:
            sliced_ranking[fold_idx][split] = {}
            for cls in ["tail", "head"]:
                sliced_ranking[fold_idx][split][cls] = {}
                for text_idx, labels_scores in ranking[fold_idx][split][cls].items():
                    sliced_ranking[fold_idx][split][cls][text_idx] = {k: v for k, v in
                                                                      heapq.nlargest(num_labels, labels_scores.items(),
                                                                                     key=lambda item: item[1])}
        return sliced_ranking

    def _fuse_rankings(self, ranking1, ranking2, fold_idx):
        fused_ranking = {
            fold_idx: {}
        }
        for split in ["train", "val", "test"]:
            fused_ranking[fold_idx][split] = {}
            for cls in ["tail", "head"]:
                r1 = ranking1[fold_idx][split][cls]
                r2 = ranking2[fold_idx][split][cls]
                fused_ranking[fold_idx][split][cls] = fuse(runs=[Run(r1), Run(r2)], norm="zmuv",
                                                           method="mnz").to_dict()

        return fused_ranking

    def _eval_ranking(self, rankings, fold_idx):
        results = []
        relevance_map = self._load_relevance_map()
        metrics = self._get_metrics()
        for split in ["train", "val", "test"]:
            for cls in ["tail", "head"]:
                ranking = rankings[fold_idx][split][cls]
                result = evaluate(
                    Qrels(
                        {key: value for key, value in relevance_map.items() if key in ranking.keys()}
                    ),
                    Run(ranking),
                    metrics
                )
                result = {k: round(v, 3) for k, v in result.items()}
                result["fold_idx"] = fold_idx
                result["split"] = split
                result["cls"] = cls
                results.append(result)
        return pd.DataFrame(results)

    def _checkpoint_rankings(self, rankings):
        with open(
                self.params.ranking.dir + self.params.model.name + "_" + self.params.data.name + ".rnk",
                "wb") as rankings_file:
            pickle.dump(rankings, rankings_file)

    def checkpoint_ranking(self, ranking, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}{self.params.model.name}_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

    def _checkpoint_results(self, results):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        pd.DataFrame(results).to_csv(
            self.params.result.dir + self.params.model.name + "_" + self.params.data.name + ".rts",
            sep='\t', index=False, header=True)

    def _checkpoint_fold_results(self, result, fold_idx):
        result_dir = f"{self.params.result.dir}{self.params.model.name}_{self.params.data.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving result for fold {fold_idx} on {result_dir}")
        pd.DataFrame(result).to_csv(
            f"{result_dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.rts",
            sep='\t', index=False, header=True)

    def _log_params_artifact(self, logger, params, artifact_name):
        # Create a temporary file to store the YAML config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            temp_file.write(OmegaConf.to_yaml(params))
            temp_file_path = temp_file.name

        # Create an artifact and add the temporary file
        artifact = wandb.Artifact(artifact_name, type='params')
        artifact.add_file(temp_file_path)

        # Log the artifact using the W&B logger
        logger.experiment.log_artifact(artifact)

    def _load_prompt(self, prompt_name):
        with open(f"{self.params.llm.dir}{self.params.data.name}/{prompt_name}.txt") as prompt_file:
            return prompt_file.read()

    def _checkpoint_prompt(self, optimized_prompt, prompt_name):
        with open(f"{self.params.llm.dir}{self.params.data.name}/{prompt_name}.txt", "w") as prompt_file:
            prompt_file.write(optimized_prompt)

    def _load_target_descriptions(self):
        with open(f"{self.params.llm.dir}{self.params.data.name}/target_descriptions.pkl",
                  "rb") as target_descriptions_file:
            return pickle.load(target_descriptions_file)
