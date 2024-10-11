import logger
log = logger.get_logger(__name__)

import torch

from typing import Callable

class FFNNModel(torch.nn.Module):

    def __init__(self,
                 n_input: int,
                 hidden_layers: list[int],
                 dropout: float,
                 n_classes: [int],
                 activation_function: Callable = torch.nn.functional.leaky_relu):
        '''
        Implements a simple feedforward neural network in PyTorch
        :param n_input: number of input features (fingerprint length)
        :param hidden_layers: array with the number of neurons in each hidden layer
        :param dropout: dropout rate
        :param n_classes: number of output classes
        :param activation_function: PyTorch activation function, e.g. torch.nn.functional.relu or torch.nn.functional.leaky_relu
        '''
        super().__init__()


        # general parameters
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.n_classes = n_classes

        # set up the linear layers
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(n_input, self.hidden_layers[0]))
        for i in range(len(self.hidden_layers)-1):
            self.linear_layers.append(torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))

        # set up the output layers, one for each task
        self.out_layers = torch.nn.ModuleList()
        for n_class in self.n_classes:
            self.out_layers.append(torch.nn.Linear(self.hidden_layers[-1], n_class))

        # dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # # initialise the weights and biases
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_uniform_(param)
        #     if 'bias' in name:
        #         torch.nn.init.constant_(param, 0.)


    def forward(self, x: torch.Tensor, task_id: int):

        # apply the linear layers, dropout after activation, https://sebastianraschka.com/faq/docs/dropout-activation.html)
        for module in self.linear_layers:
            x = self.activation_function(module(x))
            x = self.dropout(x)

        # apply the output layer (classification, produces logits)
        output = self.out_layers[task_id](x)
        return output


