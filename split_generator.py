import numpy as np
import os

input_folder = 'dataset/pcds/03467517' #Change last section to shape ID
output_folder ='dataset/splits/03467517' #Change last section to shape ID
class_id = '03467517' #Change to shape ID

def generate_poses(folder_of_pcds):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    results = []
    for filename in os.listdir(folder_of_pcds):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        results.append(f"{class_id}-{base_name}")

    train_path = os.path.join(output_folder, 'train.txt')
    test_path = os.path.join(output_folder, 'test.txt')
    val_path = os.path.join(output_folder, 'val.txt')

    for file_paths in [train_path,test_path,val_path]:
        with open(file_paths, 'w') as file:
            for line in results:
                file.write(line + '\n')

generate_poses(input_folder)