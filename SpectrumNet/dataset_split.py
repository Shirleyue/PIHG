
from create_area_dataset import read_data_path, file_filter, write_data_path
if __name__ == '__main__':

    #  --- filter the data, splitting it by frequency --------- #
    dat_lst = read_data_path('./dataset/SpectrumNet/area_test.txt')
    print('data len:', len(dat_lst))
    dic_lst = [{"frequency_id": 0}, {"frequency_id":1}, {"frequency_id":2}, {"frequency_id":3}, {"frequency_id":4}]
    scen_lst = []
    for idx, terrain_list in enumerate(dic_lst, start=0):      
        filtered_dat_lst = file_filter(dat_lst, terrain_list)
        type = terrain_list["frequency_id"]
        print(f'height{idx} len: {len(filtered_dat_lst)}')
        fre_file = f'./dataset/SpectrumNet/gen_freq_{type}.txt'
        write_data_path(filtered_dat_lst, fre_file, isShuf=False)  # 94458
    print('Done!')