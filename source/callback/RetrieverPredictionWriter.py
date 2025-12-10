import logging
from pathlib import Path
from typing import Any, List

import h5py
from pytorch_lightning.callbacks import BasePredictionWriter


def _append_to_dataset(group, key, data):
    """
    Creates the dataset if it doesn't exist, or expands it and appends the data if it already exists.
    """
    # Number of new rows in this batch
    n_new_rows = data.shape[0]

    if key not in group:
        # 1. CREATE: If it does not exist, create it with maxshape=(None, ...) to allow resizing
        # maxshape=(None, *data.shape[1:]) allows infinite growth on dimension 0
        group.create_dataset(
            key,
            data=data,
            maxshape=(None, *data.shape[1:]),
            chunks=True,  # Essential for performance and compression
            compression='gzip',
            compression_opts=4
        )
    else:
        # 2. EXPAND: If it exists, resize the dataset
        dset = group[key]
        # New size = current size + batch size
        dset.resize((dset.shape[0] + n_new_rows), axis=0)
        # Write the data at the end of the array (-n_new_rows:)
        dset[-n_new_rows:] = data


class RetrieverPredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super().__init__(write_interval=params.prediction.write_interval)
        self.params = params

        self.checkpoint_dir = (f"{self.params.prediction.dir}"
                               f"{self.params.model.name}_{self.params.data.name}/")

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.h5_path = f"{self.checkpoint_dir}{self.params.model.name}_{self.params.data.name}_{self.params.prediction.fold_idx}.h5"
        self.h5_file = None

    def on_predict_epoch_start(self, trainer, pl_module):
        self.h5_file = h5py.File(self.h5_path, 'w')

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        self.h5_file.close()
        logging.info(f"Checkpointed prediction at: {self.h5_path}.")

    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):
        modality = prediction["modality"]

        # Ensure the existence of the modality group (text or label)
        if modality not in self.h5_file:
            self.h5_file.create_group(modality)

        grp = self.h5_file[modality]

        data_to_save = {}
        if modality == "text":
            data_to_save['text_idx'] = prediction["text_idx"].cpu().numpy()
            data_to_save['text_rpr'] = prediction["text_rpr"].cpu().numpy()

        elif modality == "label":
            data_to_save['label_idx'] = prediction["label_idx"].cpu().numpy()
            data_to_save['label_rpr'] = prediction["label_rpr"].cpu().numpy()

        for key, data in data_to_save.items():
            _append_to_dataset(grp, key, data)
