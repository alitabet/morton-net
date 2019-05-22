import torch
import torch.nn as nn
from tools.utils import get_accuracy


class MortonNet(nn.Module):
    def __init__(self, input_size=3, conv_layers=3, rnn_layers=3, hidden_size=512):
        """
        MortonNet predicts the last point of a sequence of 3D points.
        The points first go through pointwise conv layers, followed
        by GRU ones. The output is the delta between the last point
        and the one before last.

        :param input_size: input feature size
        :param conv_layers: number of conv layers
        :param rnn_layers: number of GRU layers
        :param hidden_size: hidden size of GRUs
        """
        super(MortonNet, self).__init__()
        # input size is (B, S, N, 3)
        convs = nn.ModuleList()  # point feature convs
        bns = nn.ModuleList()  # respective batch norm layers
        # we create conv layers of filter size 64
        for i in range(conv_layers):
            in_layers = 64 if i > 0 else input_size
            convs.append(nn.Conv2d(in_layers, 64, kernel_size=1))
            bns.append(nn.BatchNorm2d(64))

        self.convs = convs
        self.bns = bns

        self.relu = nn.ReLU(inplace=True)

        # RNN layers aggregate all sequence points together
        self.rnn = nn.GRU(64, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)

        # fc layer to predict point displacement from the output of the RNN
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, sequences):
        # x size is (B, S, N, 3)
        x = sequences.permute(0, 3, 1, 2).contiguous()
        # x size is (B, 3, S, N)
        for i, conv in enumerate(self.convs):
            bn = self.bns[i]
            x = self.relu(bn(conv(x)))

        # x size is (B, 64, S, N)
        x = x.permute(0, 2, 3, 1).contiguous()
        # size (B, S, N, 64)
        x = x.view(-1, x.size(2), x.size(3))
        # size (BxS, N, 64)
        self.rnn.flatten_parameters()
        x, h_n = self.rnn(x)

        x = self.linear(x)

        return x, h_n


def get_point_from_orientation(sequence, out):
    """
    Given a sequence of orientations, and the output predicted
    by MortonNet, we convert the prediction into 3D points, which
    we can use for computing loss and accuracy.

    :param sequence: sequence of orientations
    :param out: output of MortonNet

    :return: sequence of points
    """
    v0 = sequence[0, :, 1, :] - sequence[0, :, 0, :]
    m0 = torch.sqrt(torch.sum(v0 ** 2, dim=-1))
    v0 /= m0.unsqueeze(-1)
    u_p = v0 + out[:, -1, :3]
    m_p = m0 + out[:, -1, -1]

    p_p = m_p.unsqueeze(-1) * u_p + sequence[0, :, -2, :]

    return p_p


def run_one_batch(model, data, criterion, optimizer, phase, loss_extent=1):
    """
    Run one batch through MortonNet network. The function takes a
    feature tensor and returns the loss and accuracy for this batch.
    If we run MortonNet in training mode, then we also compute
    gradients and update all model parameters using the optimizer.

    :param model: MortonNet model
    :param data: tuple with computed sequences and originals
    :param criterion: loss function
    :param optimizer: optimizer (if None then phase is val)
    :param phase: train, valid, or test
    :param loss_extent: number of points to include in loss computation

    :return: Batch loss and accuracy
    """
    sequences, original = data
    with torch.set_grad_enabled(phase == 'train'):
        out, _ = model(sequences[:, :, :-1, :])
        gt = sequences[0, :, -loss_extent:, :] - sequences[0, :, -loss_extent - 1:-1, :]
        loss = criterion(out[:, -loss_extent:, :].squeeze(), gt.squeeze())

        # get accuracy of point prediction
        # if we have 4 values per point then we are
        # predicted orientations, we convert those to points
        if sequences.size(-1) == 4:
            pred = get_point_from_orientation(original, out)
        else:
            pred = original[0, :, -2, :].squeeze() + out[:, -1, :]
        acc = get_accuracy(pred, original[:, :, -1, :].squeeze())

        # run backward pass during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item(), acc.item()
