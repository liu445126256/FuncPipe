import pipetests.analyzers.layer_analyzer as la
from pipetests.models.resnet import resnet101, resnet50


def split_model_by_memory(memory_dict, memory_limit):
    """
    only consider the inference memory consumes to split the model. Here simply implement the algorithm by greedy.
    :param
        memory_dict: a dict which records inference memory of each layer
        memory_limit: the upper limit of memory(MB)
            [Attention: here memory_limit only means the inference memory upper limit. The actual running memory
            consumption need extra measurement!]
    :return: split_id_list
    """
    split_model_id_list = []
    bucket_list = []
    cur_mem = 0
    for key, value in memory_dict.items():
        if cur_mem + value > memory_limit:
            bucket_list.append(cur_mem)
            split_model_id_list.append(int(key.split("--")[0]))
            cur_mem = 0
        cur_mem += value
    if cur_mem != 0:
        bucket_list.append(cur_mem)
        split_model_id_list.append(int(key.split("--")[-1]))
    print(bucket_list)  # show each split's inference memory
    split_id_list = [split_model_id_list[i + 1] - split_model_id_list[i] for i in range(len(split_model_id_list) - 1)]
    split_id_list[-1] += 1
    split_id_list.insert(0, split_model_id_list[0])
    return split_id_list


if __name__ == "__main__":
    model = resnet50()
    Flops_dict, memory_dict = la.analysis_model(model, (1, 3, 224, 224))
    print(Flops_dict)
    print(memory_dict)
    split_list = split_model_by_memory(memory_dict, 60)
    print(split_list)

    # test_dict ={
    #     '0': 5,
    #     '1': 5,
    #     '2': 5,
    #     '3': 5,
    #     '4': 5,
    #     '5': 5,
    #     '6': 5
    # }
    # split_list = split_model_by_memory(test_dict, 16)
    # print(split_list)

    # to check whether the split plan is ok:
    counter = 0
    pointer = 0
    for l in model:
        print(l)
        counter += 1
        if counter == split_list[pointer]:
            print()
            print("Here begins a New split")
            print()
            pointer += 1
            counter = 0