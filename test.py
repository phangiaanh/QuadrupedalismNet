import pickle

model_path = '/home/watermelon/sources/hcmut/QuadrupedalismNet/SMAL/smal_CVPR2018.pkl'

dd = pickle.load(open(model_path, 'rb'), encoding='latin1')

for i in dd:
    print(i)
    if isinstance(dd[i], list) or isinstance(dd[i], str):
        print(len(dd[i]))
    else:
        print(dd[i].shape)

print(dd)