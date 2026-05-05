from train import *
import hydra
import warnings
import os
from config_summary import print_config_summary

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")


@hydra.main(config_path=".", config_name="config", version_base=None)
def train(CFG):
    print_config_summary(CFG)
    train_cross_val(CFG)


if __name__ == "__main__":
    train()
