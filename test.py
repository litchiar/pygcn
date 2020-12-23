import torch
import numpy as np

# class_label={"one","two","three","four","two"}
# classes=set(class_label)
# classes_dict={c:i for i,c in enumerate(classes)}
# print(classes_dict)
# onehot=map(classes_dict.get,class_label)
# print(np.array(list(onehot)))
adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
t=adj.T>adj
print(t.dtype)
print(t.astype(np.int16))#无法直接修改
print(t)

