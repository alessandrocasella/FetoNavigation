import pandas as pd
import os
import numpy as np

general_folder_path = os.path.join(os.getcwd(), "final_dataset_files/boxplots_npy")
folder_list = sorted(os.listdir(os.path.join(general_folder_path)))

for name_folder in folder_list:
    if('.DS' in name_folder):
        folder_list.remove(name_folder)
    if('excel' in name_folder):
        folder_list.remove(name_folder)

for name_folder in folder_list:
    print(name_folder)
    file_list = sorted(os.listdir(os.path.join(general_folder_path, name_folder)))
    path_folder_files_excel = os.path.join(general_folder_path, name_folder+"_excel")
    if not os.path.exists(path_folder_files_excel):
        os.makedirs(path_folder_files_excel)

    for name_file in file_list:
        print(name_file)
        path_file = os.path.join(general_folder_path, name_folder, name_file)
        name_file_no_extension = name_file.replace('.npy', '')
        data = np.load(path_file)
        df = pd.DataFrame(data)
        filepath = os.path.join(path_folder_files_excel, name_file_no_extension+'.xlsx')
        df.to_excel(filepath, index=False)







