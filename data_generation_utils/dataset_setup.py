from semirand_pose_generator_v2 import generate_poses
from split_generator import generate_splits

class_id = '03467517'
input_folder = f'dataset/pcds/{class_id}' #Change last section to shape ID

create_splits = generate_splits(input_folder, class_id)
create_poses = generate_poses(input_folder,24,class_id)