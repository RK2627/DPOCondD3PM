"""Runs full-model offline preference training."""
from DPO.config import OfflinePreferenceConfig
from DPO.offline_preference_trainer import OfflinePreferenceTrainer


def main():
    cfg = OfflinePreferenceConfig()
    cfg.train.train_strategy = "full_model"
    trainer = OfflinePreferenceTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
