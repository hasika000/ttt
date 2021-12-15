import torch
from torch import nn

from func_lab import run_lstm_on_variable_length_seqs

m = nn.LSTM(10, 20, 1)
input = torch.randn(5, 3, 10)
h0 = torch.randn(1, 3, 20)
c0 = torch.randn(1, 3, 20)
output, (hn, cn) = m(input, (h0, c0))
print('ok')

a = run_lstm_on_variable_length_seqs(m, input)