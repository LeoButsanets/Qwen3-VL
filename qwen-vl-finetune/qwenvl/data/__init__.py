import re

DATA_1_1_0_PATH = "/mnt/DATAFAST1/multimodal/data_1.0.0"

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

ALIGNMENT_DATA = {
    "annotation_path": f"{DATA_1_1_0_PATH}/train_merged_alignment_fixed.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

INSTRUCT_DATA = {
    "annotation_path": f"{DATA_1_1_0_PATH}/train_merged_instruct_fixed.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

LLAVAMED_INSTRUCT = {
    "annotation_path": f"{DATA_1_1_0_PATH}/llavamed/train_instruct.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

RADIMAGENET_INSTRUCT = {
    "annotation_path": f"{DATA_1_1_0_PATH}/radimagenet/train_instruct.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

KITS_INSTRUCT = {
    "annotation_path": f"{DATA_1_1_0_PATH}/kits_1.0/train_instruct.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

ABDOMEN_ATLAS_INSTRUCT = {
    "annotation_path": f"{DATA_1_1_0_PATH}/abdomen_atlas_1.0/train_instruct.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

SLAKE = {
    "annotation_path": f"{DATA_1_1_0_PATH}/slake/train_instruct.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

VQA_RAD = {
    "annotation_path": f"{DATA_1_1_0_PATH}/vqa_rad/train_instruct.json",
    "data_path": f"{DATA_1_1_0_PATH}",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "alignment_data": ALIGNMENT_DATA,
    "instruct_data": INSTRUCT_DATA,
    "vqa_rad": VQA_RAD,
    "slake": SLAKE,
    "kits_instruct": KITS_INSTRUCT,
    "abdomen_atlas_instruct": ABDOMEN_ATLAS_INSTRUCT,
    "radimagenet_instruct": RADIMAGENET_INSTRUCT,
    "llavamed_instruct": LLAVAMED_INSTRUCT
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
