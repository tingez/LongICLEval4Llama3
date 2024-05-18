
from collections import defaultdict

def selectData(data_dict, label_key, shot_round=1, skip_error=True):
    """
        select several shot round data from data_dict
    """
    select_data = []
    label2_data = defaultdict(list)

    for data in data_dict:
        curr_label = data['label']
        label2_data[curr_label].append(data)

    min_count = min([len(data_list) for label, data_list in label2_data.items()])

    if not skip_error:
        assert shot_round < min_count, f'minimium count {min_count}  is less than {shot_round}'

    for idx in range(shot_round):
        for label, data_list in label2_data.items():
            if idx < len(data_list):
                select_data.append(data_list[idx])

    return select_data

