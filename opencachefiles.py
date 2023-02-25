# 两种方法都能打开
import pickle
import numpy as np

#f = open('D:/Graduation/RGAT-ABSA-master/data/output-gcn/pkls/cached_laptop_pos_tag_vocab.pkl','rb')
f = open('D:/Graduation/RGAT-ABSA-master/data/output-gcn/pkls/cached_laptop_glove_word_vocab.pkl','rb')
#f = open('D:/Graduation/RGAT-ABSA-master/data/output-gcn/pkls/cached_laptop_pos_tag_vocab.pkl','rb')
# = open('D:/Graduation/RGAT-ABSA-master/data/output-gcn/pkls/cached_laptop_pos_tag_vocab.pkl','rb')
data = pickle.load(f)
print(data)

# img_path = './train_data.pkl'
# img_data = np.load(img_path)
# print(img_data)
