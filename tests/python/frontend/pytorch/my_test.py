import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relay


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, target_size)

    def forward(self, inputs):
        lstm_out, (hidden, cell) = self.lstm(inputs)
        x = torch.cat((lstm_out[-1], cell[-1]), 1)
        logits = self.linear(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


if __name__ == "__main__":
    model = LSTM(10, 10, 10)
    torch.save(model, "./lstm.pt")
    model = model.eval()
    input_shape = [1, 5, 10]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [('input', tuple(input_shape))])
    print(mod['main'])
