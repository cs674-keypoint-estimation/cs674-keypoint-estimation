import numpy as np
import os
import random

#input_folder = 'dataset/pcds/03467517' #Change last section to shape ID
#output_folder ='dataset/splits/03467517' #Change last section to shape ID
#class_id = '03467517' #Change to shape ID

def generate_splits(folder_of_pcds, class_id):

    output_folder = f'../dataset/splits/{class_id}' #Change last section to shape ID

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    results = []
    for filename in os.listdir(folder_of_pcds):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        results.append(f"{class_id}-{base_name}")

    results = sorted(results)
    full_path = os.path.join(output_folder, 'full.txt')
    train_path = os.path.join(output_folder, 'train.txt')
    test_path = os.path.join(output_folder, 'test.txt')
    val_path = os.path.join(output_folder, 'val.txt')

    train =0.6
    test = 0.2
    val = 0.2

    with open(full_path, "w+") as f:
        for line in results:
                f.write(line + '\n')

    with open(full_path) as f:
        lines = f.readlines()
    
    #note that shuffle is in place
    random.shuffle(lines)  
    train_end_id = int(train*len(lines)+1)
    test_end_id = train_end_id+int(test*len(lines)+1)

    with open(train_path, "w") as f:
        f.writelines(lines[0:train_end_id])
    with open(test_path, "w") as f:
        f.writelines(lines[train_end_id:test_end_id])
    with open(val_path, "w") as f:
        f.writelines(lines[test_end_id:])

    """
    for file_paths in [full_path]:
        with open(file_paths, 'w') as file:
            for line in results:
                file.write(line + '\n')
    """