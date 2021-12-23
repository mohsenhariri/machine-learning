import torch
from torch.utils import data

torch.manual_seed(777)


def generate_fake_data(w, b, samples_num, pdf="normal"):
    """Generate y = w*x + b + noises"""
    m, n = w.size()
    # print(m, n)
    # mean_tensor = torch.arange(0.0, m + 1)
    # std_tensor = torch.arange(0.0, m + 1)
    # x = torch.normal(mean=mean_tensor, std=std_tensor)
    x = torch.normal(mean=0, std=1, size=(n, samples_num))
    # noise = torch.normal(0.0, 0.01, b.size())
    y = torch.matmul(w, x) + b
    # print(y)
    return x, y



# w = torch.tensor([[3, 4]], dtype=torch.float) # 12
# b = torch.tensor(1.0) # 1

w = torch.tensor([[3, 4, 4, 0],[3, 4, 4, 0]], dtype=torch.float) #24
b = torch.tensor([[1.0],[.5]]) #21



x, y = generate_fake_data(w, b, 1000)

# print(x.size())
# print(y.size())
# exit()

X = torch.t(x)
Y = torch.t(y)

dataset = data.TensorDataset(*(X, Y))
# print(dataset)

prepared_data = data.DataLoader(dataset=dataset, batch_size=5, shuffle=True)
# print(prepared_data)

data_iter = prepared_data
# data_iter = iter(prepared_data)

# print(next(data_iter))
# print(next(data_iter))


# for x,y in data_iter:
# print('y_cal',torch.matmul(w,torch.t(x)) + b)
# print('y',y)
# print(x.size())
# print(y.size())

# exit()
