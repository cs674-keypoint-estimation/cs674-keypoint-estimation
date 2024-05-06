#NOTE: NOT NEEDED, THEY SPLIT THEIR DATA AT THE ROTATION LEVEL

import random
import os 
cwd = os.getcwd()
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)
print(path)
full_filename = 'full.txt'
train_filename = 'train.txt'
test_filename = 'test.txt'
val_filename = 'val.txt'
out_file = 'out.txt'

#adapted from https://stackoverflow.com/questions/70736880/how-to-select-lines-from-a-text-file-and-then-randomize-the-order-of-those-lines

train =0.6
test = 0.2
val = 0.2



with open("full.txt") as f:
    lines = f.readlines()
#note that shuffle is in place
random.shuffle(lines)  
train_end_id = int(train*len(lines)+1)
test_end_id = train_end_id+int(test*len(lines)+1)

with open(train_filename, "w") as f:
    f.writelines(lines[0:train_end_id])
with open(test_filename, "w") as f:
    f.writelines(lines[train_end_id:test_end_id])
with open(val_filename, "w") as f:
    f.writelines(lines[test_end_id:])
#tmp=lines[n_start:n_end]
#random.shuffle(tmp)
#lines[n_start:n_end]=tmp
#with open("outfile.txt", "w") as f:
#    f.writelines(lines)
