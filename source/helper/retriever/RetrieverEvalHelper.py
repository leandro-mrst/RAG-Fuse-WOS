import logging
import pickle
from pathlib import Path

import h5py
import nmslib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

from source.helper.Helper import Helper


class RetrieverEvalHelper(Helper):
    def __init__(self, params):
        super(RetrieverEvalHelper, self).__init__()
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self._load_labels_cls()
        self.texts_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()
        self.samples_df = pd.DataFrame(self._load_samples())

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
            metrics.append(f"{metric}@{self.params.data.num_relevant_labels}")

        return metrics

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as label_cls_file:
            return pickle.load(label_cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as text_cls_file:
            return pickle.load(text_cls_file)

    def _get_split_texts_ids(self, fold_idx, split):
        split_sample_ids = self._load_split_ids(fold_idx, split)
        samples_df = self.samples_df[self.samples_df["idx"].isin(split_sample_ids)]
        return samples_df["text_idx"].to_list()

    def _get_split_labels_ids(self, fold_idx, split):
        labels_ids = set()
        split_sample_ids = self._load_split_ids(fold_idx, split)
        samples_df = self.samples_df[self.samples_df["idx"].isin(split_sample_ids)]
        for _, r in samples_df.iterrows():
            labels_ids.update(r["labels_ids"])
        return list(labels_ids)

    def _load_predictions(self, fold_idx, split):

        prediction_path = (f"{self.params.prediction.dir}{self.params.model.name}_{self.params.data.name}/"
                           f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.h5")

        split_texts_ids = np.array(self._get_split_texts_ids(fold_idx, split))
        split_labels_ids = np.array(self._get_split_labels_ids(fold_idx, split))

        text_predictions = {}
        label_predictions = {}

        with h5py.File(prediction_path, 'r') as f:
            # --- PROCESS TEXT MODALITY ---
            if "text" in f:
                # Read all IDs and Vectors into memory (fast binary read)
                all_text_ids = f["text"]["text_idx"][:]
                all_text_rpr = f["text"]["text_rpr"][:]

                # Find indices where the ID is in our split list
                # np.isin creates a boolean mask
                mask = np.isin(all_text_ids, split_texts_ids)

                if np.any(mask):
                    # Filter data using the mask
                    selected_ids = all_text_ids[mask]
                    selected_vecs = all_text_rpr[mask]

                    # Vectorized Normalization (Much faster than looping)
                    # axis=1 calculates norm across the embedding dimension
                    # keepdims=True allows broadcasting for division
                    norms = np.linalg.norm(selected_vecs, axis=1, keepdims=True)

                    # Avoid division by zero
                    norms[norms == 0] = 1e-10
                    norm_vecs = selected_vecs / norms

                    # Zip into dictionary
                    # int(k) is used to ensure Python int keys, not numpy int types
                    text_predictions = {int(k): v for k, v in zip(selected_ids, norm_vecs)}

            # --- PROCESS LABEL MODALITY ---
            if "label" in f:
                all_label_ids = f["label"]["label_idx"][:]
                all_label_rpr = f["label"]["label_rpr"][:]

                mask = np.isin(all_label_ids, split_labels_ids)

                if np.any(mask):
                    selected_ids = all_label_ids[mask]
                    selected_vecs = all_label_rpr[mask]

                    norms = np.linalg.norm(selected_vecs, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-10
                    norm_vecs = selected_vecs / norms

                    label_predictions = {int(k): v for k, v in zip(selected_ids, norm_vecs)}

        print(f"\n{split}: added {len(text_predictions)} texts")
        print(f"{split}: added {len(label_predictions)} labels\n")

        return text_predictions, label_predictions

    def init_index(self, label_predictions, cls):
        added = 0
        index = nmslib.init(method='hnsw', space='cosinesimil')
        for label_idx, label_rpr in label_predictions.items():
            if cls in self.labels_cls[label_idx]:
                index.addDataPoint(id=label_idx, data=label_rpr)
                added += 1
        # for prediction in tqdm(label_predictions, desc="Adding data to index"):
        #     if cls in self.labels_cls[prediction["label_idx"]]:
        #         index.addDataPoint(id=prediction["label_idx"], data=prediction["label_rpr"])
        #         added += 1

        index.createIndex(
            index_params=OmegaConf.to_container(self.params.eval.index),
            print_progress=False
        )
        logging.info(f"Added {added} labels.")
        return index

    def retrieve(self, index, text_predictions, cls, num_labels):
        # retrieve
        searched = 0
        ranking = {}
        index.setQueryTimeParams({'efSearch': 2048})
        for text_idx, text_rpr in tqdm(text_predictions.items(), desc="Searching"):
            if cls in self.texts_cls[text_idx]:
                retrieved_ids, distances = index.knnQuery(text_rpr, k=num_labels)
                for label_idx, distance in zip(retrieved_ids, distances):
                    if f"text_{text_idx}" not in ranking:
                        ranking[f"text_{text_idx}"] = {}
                    score = 1.0 / (distance + 1e-9)
                    if f"label_{label_idx}" in ranking[f"text_{text_idx}"]:
                        if score > ranking[f"text_{text_idx}"][f"label_{label_idx}"]:
                            ranking[f"text_{text_idx}"][f"label_{label_idx}"] = score
                    else:
                        ranking[f"text_{text_idx}"][f"label_{label_idx}"] = score
                searched += 1
        logging.info(f"Searched {searched} texts.")
        return ranking

    def _get_ranking(self, text_predictions, label_predictions, cls, num_labels):
        # index data
        index = self.init_index(label_predictions, cls)
        # retrieve
        return self.retrieve(index, text_predictions, cls, num_labels)

    def perform_eval(self):
        rankings = {}
        for fold_idx in self.params.data.folds:
            results = []
            rankings[fold_idx] = {}
            for split in ["test"]:
                rankings[fold_idx][split] = {}
                logging.info(
                    f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                    f"{OmegaConf.to_yaml(self.params)}\n")

                text_predictions, label_predictions = self._load_predictions(fold_idx, split)

                for cls in self.params.eval.label_cls:
                    logging.info(f"Evaluating {cls} ranking")
                    ranking = self._get_ranking(text_predictions, label_predictions, cls=cls,
                                                num_labels=self.params.eval.num_nearest_neighbors)
                    result = evaluate(
                        Qrels(
                            {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
                        ),
                        Run(ranking),
                        self.metrics
                    )
                    result = {k: round(v, 3) for k, v in result.items()}
                    result["fold_idx"] = fold_idx
                    result["split"] = split
                    result["cls"] = cls

                    results.append(result)
                    rankings[fold_idx][split][cls] = ranking

            self.checkpoint_ranking(rankings[fold_idx], fold_idx)
            self._checkpoint_fold_results(results, fold_idx)


