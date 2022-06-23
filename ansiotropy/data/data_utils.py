from cProfile import label
from openprompt.data_utils import PROCESSORS

DATA_DIR = "./data/"

SUPERGLUE_TASKS = [
    "multirc",
    "boolq",
    "cb",
    "copa",
    "rte",
    "wic",
    "wsc",
    "record",
]


SUPERGLUE_SCRIPTS_BASE = {
    "boolq": "SuperGLUE/BoolQ",
    "copa": "SuperGLUE/CoPA",
    "multirc": "SuperGLUE/MultiRC",
    "stsb": "SuperGLUE/STS-B",
    "wsc": "SuperGLUE/WSC",
    "rte": "SuperGLUE/RTE",
    "mnli": "SuperGLUE/MNLI",
    "mrpc": "SuperGLUE/MRPC",
    "sst2": "SuperGLUE/SST-2",
    "wic": "SuperGLUE/WIC",
    "qqp": "SuperGLUE/QQP",
    "qnli": "SuperGLUE/QNLI",
    "mnli_mm": "SuperGLUE/MNLI-MM",
    "cb": "SuperGLUE/CB", 
}


def load_validation_data(dataset: str):
    if dataset in SUPERGLUE_TASKS:
        dataset = "super_glue.{}".format(dataset)
    Processor = PROCESSORS[dataset]
    validation_dataset = Processor().get_dev_examples(DATA_DIR)
    class_labels = Processor().get_labels()
    return validation_dataset, class_labels
