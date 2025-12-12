import logging

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from source.datamodule.RetrieverDataModule import RetrieverDataModule
from source.helper.Helper import Helper
from source.model.RetrieverModel import RetrieverModel


class RetrieverFitHelper(Helper):
    def __init__(self, params):
        super(RetrieverFitHelper, self).__init__()
        self.params = params
        logging.basicConfig(level=logging.INFO)

    def run(self):
        seed_everything(707, workers=True)
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Fitting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"self.params\n {OmegaConf.to_yaml(self.params)}\n")

            logger = self.get_logger(fold_idx)
            self._log_params_artifact(logger, self.params, artifact_name="fit_params")

            # Initialize a trainer
            trainer = pl.Trainer(
                accelerator=self.params.trainer.accelerator,
                devices=self.params.trainer.devices,
                max_epochs=self.params.trainer.max_epochs,
                precision=self.params.trainer.precision,
                logger=logger,
                callbacks=[
                    self._get_model_checkpoint_callback(fold_idx, monitor=self.params.trainer.monitor, mode=self.params.trainer.mode),
                    self._get_early_stopping_callback(monitor=self.params.trainer.monitor, mode=self.params.trainer.mode),
                    self._get_lr_monitor(),
                    self._get_progress_bar_callback()
                ]
            )

            # datamodule
            datamodule = RetrieverDataModule(
                params=self.params.data,
                tokenizer=self.get_tokenizer(),
                fold_idx=fold_idx)

            # model
            model = RetrieverModel(self.params.model)

            # Train the ⚡ model
            trainer.fit(
                model=model,
                datamodule=datamodule
            )
