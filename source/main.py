import os
from entrance import UTHE, EUTHE, FUTHE, FEUTHC, AUTHE
from config import EUTHC, FUTHC, UTHC, FEUTHC, AUTHC

if __name__ ==  "__main__":
    config = EUTHC
    config.model_path = os.path.join(config.project_dir, "output/model/EUTH/EUTH_lstm.pkl")
    entrance = EUTHE()
    entrance.train()