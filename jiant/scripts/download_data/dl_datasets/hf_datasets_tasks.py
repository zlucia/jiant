"""Use this for tasks that can be obtained from HF-Datasets without further/special processing"""

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.io as py_io

# Note to future selves: beware of circular imports when refactoring
from jiant.tasks.retrieval import (
    ColaTask,
    MrpcTask,
    QnliTask,
    QqpTask,
    RteTask,
    SstTask,
    WnliTask,
    BoolQTask,
    CommitmentBankTask,
    WiCTask,
    WSCTask,
    SuperglueWinogenderDiagnosticsTask,
    GlueDiagnosticsTask,
)


HF_DATASETS_CONVERSION_DICT = {
    # === GLUE === #
    "cola": {
        "path": "glue",
        "name": "cola",
        "field_map": {"sentence": "text"},
        "label_map": ColaTask.ID_TO_LABEL,
    },
    "mnli": {
        "path": "glue",
        "name": "mnli",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "phase_map": {"validation_matched": "val", "test_matched": "test"},
        "phase_list": ["train", "val", "test"],
    },
    "mnli_mismatched": {
        "path": "glue",
        "name": "mnli",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "phase_map": {"validation_mismatched": "val", "test_mismatched": "test"},
        "phase_list": ["val", "test"],
        "jiant_task_name": "mnli_mismatched",
    },
    "mrpc": {
        "path": "glue",
        "name": "mrpc",
        "field_map": {"sentence1": "text_a", "sentence2": "text_b"},
        "label_map": MrpcTask.ID_TO_LABEL,
    },
    "qnli": {
        "path": "glue",
        "name": "qnli",
        "field_map": {"question": "premise", "sentence": "hypothesis"},
        "label_map": QnliTask.ID_TO_LABEL,
    },
    "qqp": {
        "path": "glue",
        "name": "qqp",
        "field_map": {"question1": "text_a", "question2": "text_b"},
        "label_map": QqpTask.ID_TO_LABEL,
    },
    "rte": {
        "path": "glue",
        "name": "rte",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": RteTask.ID_TO_LABEL,
    },
    "sst": {
        "path": "glue",
        "name": "sst2",
        "field_map": {"sentence": "text"},
        "label_map": SstTask.ID_TO_LABEL,
    },
    "stsb": {
        "path": "glue",
        "name": "stsb",
        "field_map": {"sentence1": "text_a", "sentence2": "text_b"},
    },
    "wnli": {
        "path": "glue",
        "name": "wnli",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": WnliTask.ID_TO_LABEL,
    },
    "glue_diagnostics": {
        "path": "glue",
        "name": "ax",
        "label_map": GlueDiagnosticsTask.ID_TO_LABEL,
        "phase_map": None,
        "jiant_task_name": "glue_diagnostics",
    },
    # === SuperGLUE === #
    "boolq": {"path": "super_glue", "name": "boolq", "label_map": BoolQTask.ID_TO_LABEL},
    "cb": {"path": "super_glue", "name": "cb", "label_map": CommitmentBankTask.ID_TO_LABEL},
    "copa": {"path": "super_glue", "name": "copa"},
    "multirc": {"path": "super_glue", "name": "multirc"},
    "record": {"path": "super_glue", "name": "record"},
    "wic": {"path": "super_glue", "name": "wic", "label_map": WiCTask.ID_TO_LABEL},
    "wsc": {"path": "super_glue", "name": "wsc.fixed", "label_map": WSCTask.ID_TO_LABEL},
    "superglue_broadcoverage_diagnostics": {
        "path": "super_glue",
        "name": "axb",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": RteTask.ID_TO_LABEL,
        "phase_map": None,
        "jiant_task_name": "rte",
    },
    "superglue_winogender_diagnostics": {
        "path": "super_glue",
        "name": "axg",
        "label_map": SuperglueWinogenderDiagnosticsTask.ID_TO_LABEL,
        "phase_map": None,
        "jiant_task_name": "superglue_axg",
    },
    # === Other === #
    "snli": {"path": "snli", "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"}},
    "commonsenseqa": {"path": "commonsense_qa", "phase_list": ["train", "val", "test"]},
    "hellaswag": {
        "path": "hellaswag",
        "phase_list": ["train", "val", "test"],
        "label_map": {"0": 0, "1": 1, "2": 2, "3": 3},
    },
    "cosmosqa": {"path": "cosmos_qa", "phase_list": ["train", "val", "test"]},
    "socialiqa": {"path": "social_i_qa", "phase_list": ["train", "val"]},
    "scitail": {"path": "scitail", "name": "tsv_format", "phase_list": ["train", "val", "test"]},
    "quoref": {"path": "quoref", "phase_list": ["train", "val"]},
    "adversarial_nli_r1": {
        "path": "anli",
        "field_map": {"premise": "context"},
        "label_map": {0: "e", 1: "n", 2: "c"},
        "phase_map": {"train_r1": "train", "dev_r1": "val", "test_r1": "test"},
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "adversarial_nli",
    },
    "adversarial_nli_r2": {
        "path": "anli",
        "field_map": {"premise": "context"},
        "label_map": {0: "e", 1: "n", 2: "c"},
        "phase_map": {"train_r2": "train", "dev_r2": "val", "test_r2": "test"},
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "adversarial_nli",
    },
    "adversarial_nli_r3": {
        "path": "anli",
        "field_map": {"premise": "context"},
        "label_map": {0: "e", 1: "n", 2: "c"},
        "phase_map": {"train_r3": "train", "dev_r3": "val", "test_r3": "test"},
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "adversarial_nli",
    },
    "arc_easy": {
        "path": "ai2_arc",
        "name": "ARC-Easy",
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "arc_easy",
    },
    "arc_challenge": {
        "path": "ai2_arc",
        "name": "ARC-Challenge",
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "arc_challenge",
    },
    "race": {
        "path": "race",
        "name": "all",
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "race",
    },
    "race_middle": {
        "path": "race",
        "name": "middle",
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "race",
    },
    "race_high": {
        "path": "race",
        "name": "high",
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "race",
    },
    "quail": {
        "path": "quail",
        "phase_list": ["train", "val", "test"],
        "jiant_task_name": "quail",
        "phase_map": {"validation": "val", "challenge": "test"},
    },
    "casehold": {
        "path": "zlucia/casehold_aux",
        "phase_list": ["train"],
        "phase_map": None,
        "label_map": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
        "jiant_task_name": "casehold"
    },
    "hendrycks_formal_logic": {
        "path": "zlucia/hendrycks_test_select",
        "name": "formal_logic",
        "phase_list": ["train"],
        "phase_map": None,
        "jiant_task_name": "hendrycks_test_select"
    },
    "hendrycks_hs_gov_politics": {
        "path": "zlucia/hendrycks_test_select",
        "name": "high_school_government_and_politics",
        "phase_list": ["train"],
        "phase_map": None,
        "jiant_task_name": "hendrycks_test_select"
    },
    "hendrycks_intl_law": {
        "path": "zlucia/hendrycks_test_select",
        "name": "international_law",
        "phase_list": ["train"],
        "phase_map": None,
        "jiant_task_name": "hendrycks_test_select"
    },
    "hendrycks_jurisprudence": {
        "path": "zlucia/hendrycks_test_select",
        "name": "jurisprudence",
        "phase_list": ["train"],
        "phase_map": None,
        "jiant_task_name": "hendrycks_test_select"
    },
    "hendrycks_logical_fallacies": {
        "path": "zlucia/hendrycks_test_select",
        "name": "logical_fallacies",
        "phase_list": ["train"],
        "phase_map": None,
        "jiant_task_name": "hendrycks_test_select"
    },
    "hendrycks_moral_disputes": {
        "path": "zlucia/hendrycks_test_select",
        "name": "moral_disputes",
        "phase_list": ["train"],
        "phase_map": None,
        "jiant_task_name": "hendrycks_test_select"
    },
    "hendrycks_philosophy": {
        "path": "zlucia/hendrycks_test_select",
        "name": "philosophy",
        "phase_list": ["train"],
        "phase_map": None,
        "jiant_task_name": "hendrycks_test_select"
    },
    "mbe_fold_1_val": {
        "path": "zlucia/mbe",
        "name": "fold_1_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_1_test": {
        "path": "zlucia/mbe",
        "name": "fold_1_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_2_val": {
        "path": "zlucia/mbe",
        "name": "fold_2_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_2_test": {
        "path": "zlucia/mbe",
        "name": "fold_2_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_3_val": {
        "path": "zlucia/mbe",
        "name": "fold_3_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_3_test": {
        "path": "zlucia/mbe",
        "name": "fold_3_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_4_val": {
        "path": "zlucia/mbe",
        "name": "fold_4_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_4_test": {
        "path": "zlucia/mbe",
        "name": "fold_4_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_5_val": {
        "path": "zlucia/mbe",
        "name": "fold_5_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_5_test": {
        "path": "zlucia/mbe",
        "name": "fold_5_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_6_val": {
        "path": "zlucia/mbe",
        "name": "fold_6_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_6_test": {
        "path": "zlucia/mbe",
        "name": "fold_6_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_7_val": {
        "path": "zlucia/mbe",
        "name": "fold_7_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_7_test": {
        "path": "zlucia/mbe",
        "name": "fold_7_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_8_val": {
        "path": "zlucia/mbe",
        "name": "fold_8_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_8_test": {
        "path": "zlucia/mbe",
        "name": "fold_8_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_9_val": {
        "path": "zlucia/mbe",
        "name": "fold_9_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_9_test": {
        "path": "zlucia/mbe",
        "name": "fold_9_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_10_val": {
        "path": "zlucia/mbe",
        "name": "fold_10_val",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    },
    "mbe_fold_10_test": {
        "path": "zlucia/mbe",
        "name": "fold_10_test",
        "phase_list": ["train", "val"],
        "jiant_task_name": "mbe"
    }
}

# HF-Datasets uses "validation", we use "val"
DEFAULT_PHASE_MAP = {"validation": "val"}


def download_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    hf_datasets_conversion_metadata = HF_DATASETS_CONVERSION_DICT[task_name]
    examples_dict = download_utils.convert_hf_dataset_to_examples(
        path=hf_datasets_conversion_metadata["path"],
        name=hf_datasets_conversion_metadata.get("name"),
        field_map=hf_datasets_conversion_metadata.get("field_map"),
        label_map=hf_datasets_conversion_metadata.get("label_map"),
        phase_map=hf_datasets_conversion_metadata.get("phase_map", DEFAULT_PHASE_MAP),
        phase_list=hf_datasets_conversion_metadata.get("phase_list"),
    )
    paths_dict = download_utils.write_examples_to_jsonls(
        examples_dict=examples_dict, task_data_path=task_data_path,
    )
    jiant_task_name = hf_datasets_conversion_metadata.get("jiant_task_name", task_name)
    py_io.write_json(
        data={"task": jiant_task_name, "paths": paths_dict, "name": task_name},
        path=task_config_path,
    )
