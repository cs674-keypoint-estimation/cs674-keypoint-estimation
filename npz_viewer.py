from numpy import load


link1 = 'test1/1b300fd9ad4050e6301fa0a0663ee996.npz'
link2 = 'dataset/poses/02691156/1b300fd9ad4050e6301fa0a0663ee996.npz'
data = load(link1)
lst = data.files
for item in lst:
    print(item)
    print(data[item])
