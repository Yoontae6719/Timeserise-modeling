from .LSTM import LSTM
from .utils import G_loss_fun
from .distribution import Gaussian, NegativeBinomial
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, raw_params):
        super(Net, self).__init__()

        self.params = dict(raw_params) # copy

        # Network parameters
        self.embedding = nn.Embedding(self.params["num_class"], self.params["embedding_dim"] )

        self.lstm = LSTM(input_size= 1 + self.params["cov_dim"] + self.params["embedding_dim"],
                         hidden_size= self.params["lstm_hidden_dim"],
                         num_layers= self.params["lstm_layers"],
                         bias = True,
                         batch_first = False,
                         dropout= self.params["lstm_dropout"])

        self.relu = nn.ReLU()
        if self.params["distributon"] == "g":
            self.likelihood = Gaussian(self.params["lstm_hidden_dim"] * self.params["lstm_layers"], 1)
        elif self.params["distribution"] == "negbin":
            self.likelihood = NegativeBinomial()
        else:
            print("Set distribution!!")

    def forward(self, x, idx, hidden, cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''

        onehot_encoding = self.embedding(idx) # 370 to 20

        lstm_input = torch.cat((x, onehot_encoding), dim = 2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1) #batch_size, lstm_hidden_dimn, lstm_layers
        mu, sigma = self.likelihood(hidden_permute)

        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params["lstm_layers"],
                           input_size,
                           self.params["lstm_hidden_dim"],
                           device=self.params["device"])

    def init_cell(self, input_size):
        return torch.zeros(self.params["lstm_layers"],
                           input_size,
                           self.params["lstm_hidden_dim"],
                           device=self.params["device"])
