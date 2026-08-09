
import random
from utils import read_data_path
from create_area_dataset import file_filter

def write_data_path(dataList, fileName, isShuf=False):
    # dataList: a list of data paths and tags,
    # fileName: the txt file name to write path to
    # isShuf: Indicates whether the data set is shuffled
    if isShuf:
        random.shuffle(dataList)
    with open(fileName, 'w', encoding='UTF-8') as f:
        for dat in dataList:
            f.write(str(dat))

if __name__ == "__main__":
    #   ------ merge----------------  #
    scen = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'scenario_6',
            'scenario_7', 'scenario_8', 'scenario_9', 'scenario_10', 'scenario_11']
    total_name = 'area'
    total_train_lst = []
    for s in scen:
        train_txt = './dataset/SpectrumNet/' + s + ' _train.txt'
        train_lst = read_data_path(train_txt)
        total_train_lst += train_lst
    print('len(total_train_lst)', len(total_train_lst))
    write_data_path(total_train_lst, './dataset/SpectrumNet/' + total_name + ' _train.txt', isShuf=True)



    scen = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'scenario_6',
            'scenario_7', 'scenario_8', 'scenario_9', 'scenario_10', 'scenario_11']
    total_name = 'area'
    total_valid_lst = []
    for s in scen:
        valid_txt = './dataset/SpectrumNet/' + s + ' _valid.txt'
        valid_lst = read_data_path(valid_txt)
        total_valid_lst += valid_lst
    print('len(total_valid_lst)', len(total_valid_lst))
    write_data_path(total_valid_lst, './dataset/SpectrumNet/' + total_name + ' _valid.txt', isShuf=False)

    scen = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'scenario_6',
            'scenario_7', 'scenario_8', 'scenario_9', 'scenario_10', 'scenario_11']
    total_name = 'area'
    total_test_lst = []
    for s in scen:
        test_txt = './dataset/SpectrumNet/' + s + ' _test.txt'
        test_lst = read_data_path(test_txt)
        total_test_lst += test_lst
    print('len(total_test_lst)', len(total_test_lst))
    write_data_path(total_test_lst, './dataset/SpectrumNet/' + total_name + ' _test.txt', isShuf=True)

    #   ------ frequency ----------------  #

    dat_lst = read_data_path('./dataset/SpectrumNet/area _train.txt')
    print('data len:', len(dat_lst))
    dic_lst = [{"frequency_id": 0}]
    scen_lst = []
    for idx, terrain_list in enumerate(dic_lst, start=0):
        filtered_dat_lst = file_filter(dat_lst, terrain_list)
        scen_lst.extend(filtered_dat_lst)
    print(f'freq train len: {len(scen_lst)}')
    write_data_path(scen_lst, './dataset/SpectrumNet/freq _train.txt', isShuf=True)

    dat_lst = read_data_path('./dataset/SpectrumNet/area _valid.txt')
    print('data len:', len(dat_lst))
    dic_lst = [{"frequency_id": 3}, {"frequency_id": 1}, {"frequency_id": 2}, {"frequency_id":4}]
    scen_lst = []
    for idx, terrain_list in enumerate(dic_lst, start=0):
        filtered_dat_lst = file_filter(dat_lst, terrain_list)
        scen_lst.extend(filtered_dat_lst)
    print(f'freq valid len: {len(scen_lst)}')
    write_data_path(scen_lst, './dataset/SpectrumNet/freq _valid.txt', isShuf=False)
    print('Done!')

    dat_lst = read_data_path('./dataset/SpectrumNet/area _test.txt')
    print('data len:', len(dat_lst))
    dic_lst = [{"frequency_id": 3}, {"frequency_id": 1}, {"frequency_id": 2}, {"frequency_id":4}]
    scen_lst = []
    for idx, terrain_list in enumerate(dic_lst, start=0):
        filtered_dat_lst = file_filter(dat_lst, terrain_list)
        scen_lst.extend(filtered_dat_lst)
    print(f'freq test len: {len(scen_lst)}')
    write_data_path(scen_lst, './dataset/SpectrumNet/freq _test.txt', isShuf=False)
    print('Done!')

    print('Done!')
