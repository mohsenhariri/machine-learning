import torch
# from torch.serialization import load


def save_model(model,path):
  torch.save(model,path)


# print(model.eval()) 


# model = Model(*args,**kwargs)

# model.load_stat

model = torch.load('./mnist/model/21-35-46.pthfull')
# print(model.eval())


