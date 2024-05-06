import numpy as np
import os

input_folder = 'dataset/pcds/03467517' #Change last section to shape ID
output_folder ='dataset/poses/03467517' #Change last section to shape ID

def generate_camera_mat():
    return np.array([
        [149.84375, 0.0, 68.5],
        [0.0, 149.84375, 68.5],
        [0.0, 0.0, 1.0]
    ])

def generate_world_mat(filename):

    base_name = os.path.splitext(os.path.basename(filename))[0]

    #print(base_name)

    random_matrix = np.random.rand(3,4)

    U, D, V = np.linalg.svd(random_matrix, full_matrices=False)

    Vt = V.transpose(0,1)

    world_matrix = np.dot(U, Vt)

    #print(world_matrix)
    npz_filename = f"{base_name}.npz"
    return npz_filename, world_matrix
    #np.savez(npz_filename, _world_mat_=world_matrix)

def generate_poses(folder_of_pcds,rotations):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    poses_list ={}
    for filename in os.listdir(folder_of_pcds):
        for index in range(rotations):
            #print("index: ",index)
            #print("name: ",filename)


            camera_mat = generate_camera_mat()
            npz_name, world_mat = generate_world_mat(filename)

            poses_list[f'camera_mat_{index}'] = camera_mat
            poses_list[f'world_mat_{index}'] = world_mat
        new_filename = os.path.splitext(filename)[0]
        npz_path = os.path.join(output_folder, f"{new_filename}.npz")
        np.savez(npz_path, **poses_list)
        #print(poses_list)
    return None     

#print(os.listdir('dataset/pcds'))
generate_poses(input_folder, 24)

