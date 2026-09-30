import os
from typing import Optional

import hydra
import transformers
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from autoindexer.utils.trainer import build_trainer
from autoindexer.utils.hydra_utils import set_sys_arg_defaults_for_hydra


def train_func(hydra_config):
    trainer, config, dataset = build_trainer(hydra_config)

    if config.mode == "train":
        trainer.train(resume_from_checkpoint=config.get("checkpoint"))
    elif config.mode == "eval":
        trainer.evaluate()
    elif config.mode == "test":
        trainer.predict(dataset["test"])
    else:
        raise ValueError(f"Unknown mode {config.mode}")
    return trainer.model


@hydra.main(version_base=None)
def main(hydra_config: DictConfig) -> None:
    if hydra_config.get("seed_everything") is not None:
        transformers.set_seed(hydra_config.seed_everything)
    else:
        print("Warning: No seed specified. Results may be non-deterministic.")

    print("=== Config ===")
    print(OmegaConf.to_yaml(hydra_config))
    print("==============")
    print()

    model = train_func(hydra_config)
    print(model)


if __name__ == "__main__":
    HF_ACCESS_TOKEN = os.environ.get("HF_TOKEN")
    if HF_ACCESS_TOKEN:
        login(token=HF_ACCESS_TOKEN)

    set_sys_arg_defaults_for_hydra(default_config_path="configs/cpt_datamix/qwen3_8B_lora_p5.yaml")
    main()
