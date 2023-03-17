import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        # self.norm = nn.BatchNorm1d(output_dim)
        self.act = act
    
    def forward(self, x):
        return self.act(self.dense(x))


class DNN(nn.Module):
    def __init__(self, dim_list):
        super(DNN, self).__init__()
        self.layers = nn.Sequential()

        for i in range(len(dim_list)-1):
            self.layers.add_module(f'mlp{i}', MLP(dim_list[i], dim_list[i+1]))
        
        self.layers.add_module(f'fc', nn.Linear(dim_list[-1], 1))

    def forward(self, x):
        return self.layers(x).squeeze(1)  # (B, 1) -> (B)


if __name__ == "__main__":
    model = DNN([8, 16, 32, 64])
    print(model)
