import logging

import pytorch_lightning as pl
from omegaconf import OmegaConf

from source.callback.RetrieverPredictionWriter import RetrieverPredictionWriter
from source.datamodule.RetrieverDataModule import RetrieverDataModule
from source.helper.Helper import Helper
from source.model.RetrieverModel import RetrieverModel


class RetrieverPredictHelper(Helper):


    def __init__(self, params):
        super(RetrieverPredictHelper, self).__init__()
        self.params = params
        logging.basicConfig(level=logging.INFO)

    def perform_predict(self):
        for fold_idx in self.params.data.folds:
            # data
            datamodule = RetrieverDataModule(
                self.params.data,
                tokenizer=self.get_tokenizer(),
                fold_idx=fold_idx)

            # model
            if not self.params.model.zero_shot:
                model = RetrieverModel.load_from_checkpoint(
                    checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}/"
                                    f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.ckpt"
                )
                logging.info(f"Predicting using {self.params.model.name} fine-tuned.")


            else:
                model = RetrieverModel(self.params.model)
                logging.info(f"Predicting using zero-shot {self.params.model.name}.")

            self.params.prediction.fold_idx = fold_idx
            # trainer
            trainer = pl.Trainer(
                accelerator=self.params.trainer.accelerator,
                devices=self.params.trainer.devices,
                max_epochs=self.params.trainer.max_epochs,
                precision=self.params.trainer.precision,
                callbacks=[RetrieverPredictionWriter(self.params)]
            )

            # predicting
            datamodule.prepare_data()
            datamodule.setup("predict")

            logging.info(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.predict(
                model=model,
                datamodule=datamodule,

            )
