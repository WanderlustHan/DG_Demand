import math
from typing import List, Tuple

import torch

torch.cuda.current_device()
from torch import nn, Tensor
import torch.nn.functional as F


class MSDRCell(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, pre_k: int, pre_v: int, num_nodes: int, n_supports: int,
                 k_hop: int, e_layer: int, n_dim: int):
        super(MSDRCell, self).__init__()
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.pre_v = pre_v
        self.W = nn.Parameter(torch.zeros(hidden_size*2, hidden_size*2), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_nodes, hidden_size*2), requires_grad=True)
        self.R = nn.Parameter(torch.zeros(pre_k, num_nodes, hidden_size), requires_grad=True)
        self.attlinear = nn.Linear(num_nodes * hidden_size*2, 1)
        self.evolution = EvolutionCell(input_dim*2 + hidden_size*2, hidden_size*2, num_nodes, n_supports, k_hop,
                                       e_layer, n_dim)
        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([12.0]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_size]),
            requires_grad=False
        )

    def forward(self, inputs: Tensor, supports: Tensor, hidden_states) -> Tuple[Tensor, Tensor]:
        bs, k, n, d = hidden_states.size()
        _, _, f = inputs.size()
        re_inputs, im_inputs = torch.chunk(inputs, 2, dim=-1)
        preH = hidden_states[:, -1:]
        preH = preH.reshape(bs, n, d)
        re_preH, im_preH = torch.chunk(preH, 2, dim=-1)
        re_convInput = torch.cat([re_inputs, re_preH], dim=-1)
        im_convInput = torch.cat([im_inputs, im_preH], dim=-1)
        convInput = torch.cat([re_convInput, im_convInput], dim=-1)
        convOutput = F.leaky_relu_(self.evolution(convInput, supports))

        pi = 3.14159265358979323846
        phase_relation = self.R / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        hd = hidden_states + 0.0
        # 得到hidden_states的实数部分re和虚数部分im。hidden_states就是head
        re_hidden_states, im_hidden_states = torch.chunk(hd, 2, dim=-1)
        # 分别计算tail = head * relation 的实部与虚部。公式来源于RotatE代码
        re_new_states = re_relation*re_hidden_states - im_relation*im_hidden_states
        im_new_states = im_relation*re_hidden_states + re_relation*im_hidden_states
        # 将新的实部虚部组合在一起
        new_states = torch.cat([re_new_states, im_new_states], dim=-1)
        output = torch.matmul(convOutput, self.W) + self.b.unsqueeze(0) + self.attention(new_states)
        output = output.reshape(bs, 1, n, d)
        x = hidden_states[:, 1:k]
        hidden_states = torch.cat([x, output], dim=1)
        output = output.reshape(bs, n, d)
        return output, hidden_states

    def attention(self, inputs: Tensor):
        bs, k, n, d = inputs.size()
        x = inputs.reshape(bs, k, -1)
        out = self.attlinear(x)
        weight = F.softmax(out, dim=1)
        outputs = (x * weight).sum(dim=1).reshape(bs, n, d)
        return outputs


class GMSDREncoder(nn.ModuleList):
    def __init__(self, input_size: int, hidden_size: int, pre_k: int, pre_v, num_node: int, n_supports: int, k_hop: int,
                 n_layers: int, e_layer: int, n_dim: int):
        super(GMSDREncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.append(MSDRCell(input_size, hidden_size, pre_k, pre_v, num_node, n_supports, k_hop, e_layer, n_dim))
        for _ in range(1, n_layers):
            self.append(MSDRCell(hidden_size, hidden_size, pre_k, pre_v, num_node, n_supports, k_hop, e_layer, n_dim))
        self.pre_k = pre_k

    def forward(self, inputs: Tensor, supports: List[Tensor]):
        bs, lens, n, f = inputs.shape
        dv, dt = inputs.device, inputs.dtype
        inputs = list(inputs.transpose(0, 1))
        states = []
        for i_t in range(self.n_layers):
            states.append(torch.zeros(bs, self.pre_k, n, self.hidden_size*2, device=dv, dtype=dt))
        outputs = inputs
        for i_layer, cell in enumerate(self):
            for i_t in range(lens):
                outputs[i_t], states[i_layer] = cell(inputs[i_t], supports, states[i_layer])
        return outputs, torch.stack(states)


class GMSDRDecoder(nn.ModuleList):
    def __init__(self, output_size: int, hidden_size: int, pre_k: int, pre_v, num_node: int,
                 n_supports: int, k_hop: int, n_layers: int, n_preds: int, e_layer: int, n_dim: int):
        super(GMSDRDecoder, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.append(MSDRCell(hidden_size, hidden_size, pre_k, pre_v, num_node, n_supports, k_hop, e_layer, n_dim))
        for _ in range(1, n_layers):
            self.append(MSDRCell(hidden_size, hidden_size,pre_k, pre_v, num_node, n_supports, k_hop, e_layer, n_dim))
        self.out = nn.Linear(hidden_size*2, output_size)

    def forward(self, inputs: Tensor, supports: List[Tensor], states):
        lens = len(inputs)
        outputs = []
        for i_t in range(lens):
            for i_layer in range(self.n_layers):
                inputs[i_t], states[i_layer] = self[i_layer](inputs[i_t], supports, states[i_layer])
                if (i_layer == self.n_layers - 1):
                    output = self.out(inputs[i_t])
                    outputs.append(output)
        outputs = torch.stack(outputs, 1)
        outputs, _ = torch.chunk(outputs, 2, dim=-1)
        return outputs


class EvolutionCell(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int, layer: int,
                 n_dim: int):
        super(EvolutionCell, self).__init__()
        self.layer = layer
        self.input_dim = input_dim
        self.perceptron = nn.ModuleList()
        self.graphconv = nn.ModuleList()
        self.attlinear = nn.Linear(num_nodes * output_dim, 1)
        self.graphconv = GraphConv_(input_dim, output_dim, num_nodes, n_supports, max_step)
        # for i in range(1, layer):
        #     self.graphconv.append(GraphConv_(output_dim, output_dim, num_nodes, n_supports, max_step))

    def forward(self, inputs, supports: List):
        outputs = []
        supports = torch.stack(supports, 0)
        # for i in range(self.layer):
        inputs = self.graphconv(inputs, [supports])
        outputs.append(inputs)
        out = self.attention(torch.stack(outputs, dim=1))
        return out

    def attention(self, inputs: Tensor):
        b, g, n, f = inputs.size()
        x = inputs.reshape(b, g, -1)
        out = self.attlinear(x)
        weight = F.softmax(out, dim=1)
        outputs = (x * weight).sum(dim=1).reshape(b, n, f)
        return outputs


class GraphConv_(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(GraphConv_, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(input_dim * num_metrics, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):

        b, n, input_dim = inputs.shape  # [batch, node, hid_dim]
        x = inputs
        # x0 = x.permute([1, 2, 0]).reshape(n, -1) # [nodes, hid_dim*batch_size]
        x0 = x
        x = x0.unsqueeze(dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.matmul(support,x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.matmul(support,x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        # x = x.reshape(-1, n, input_dim, b).transpose(0, 3) # 用于单个graph
        x = x.permute(1,2,3,0) # 动态graph，需注释上一句
        x = x.reshape(b, n, -1)
        #  2 x = [batch, node, hid_dim*(k_hop+1)] == 1
        return self.out(x)


class GMSDRNet(nn.Module):
    def __init__(self,
                 lens: int,
                 pre_k: int,
                 pre_v: int,
                 hidden_size: int,
                 num_nodes: int,
                 n_dim: int,
                 n_supports: int,
                 k_hop: int,
                 n_rnn_layers: int,
                 n_gconv_layers: int,
                 input_dim: int,
                 output_dim: int,
                 cl_decay_steps: int,
                 support,
                 device):
        super(GMSDRNet, self).__init__()
        self.cl_decay_steps = cl_decay_steps
        # self.lens = lens
        self.device = device
        self.support = support
        # self.num_nodes = num_nodes
        # self.output_dim = output_dim
        # m, p, n = torch.svd(support)
        # initemb1 = torch.mm(m[:, :n_dim], torch.diag(p[:n_dim] ** 0.5))
        # initemb2 = torch.mm(torch.diag(p[:n_dim] ** 0.5), n[:, :n_dim].t())
        #
        # self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
        # self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
        #
        # self.n_gconv_layers = n_gconv_layers
        self.encoder = GMSDREncoder(input_dim, hidden_size, pre_k, pre_v, num_nodes, n_supports, k_hop, n_rnn_layers,
                                    n_gconv_layers, n_dim)
        self.decoder = GMSDRDecoder(output_dim*2, hidden_size, pre_k, pre_v, num_nodes, n_supports, k_hop, n_rnn_layers,
                                    pre_k, n_gconv_layers, n_dim)
        # self.w1 = nn.Parameter(torch.eye(n_dim), requires_grad=True)
        # self.w2 = nn.Parameter(torch.eye(n_dim), requires_grad=True)
        # self.b1 = nn.Parameter(torch.zeros(n_dim), requires_grad=True)
        # self.b2 = nn.Parameter(torch.zeros(n_dim), requires_grad=True)
        #
        # self.graph0 = None
        # self.graph1 = None
        # self.graph2 = None

    def forward(self, inputs: Tensor, supports) :
        graph = list(supports)
        # graph = list(self.support)
        # nodevec1 = self.nodevec1
        # nodevec2 = self.nodevec2
        # n = nodevec1.size(0)
        # self.graph0 = F.leaky_relu_(torch.mm(nodevec1, nodevec2))
        # graph.append(self.graph0)
        #
        # nodevec1 = nodevec1.mm(self.w1) + self.b1.repeat(n, 1)
        # nodevec2 = (nodevec2.T.mm(self.w1) + self.b1.repeat(n, 1)).T
        # self.graph1 = F.leaky_relu_(torch.mm(nodevec1, nodevec2))
        # graph.append(self.graph1)
        #
        # nodevec1 = nodevec1.mm(self.w2) + self.b2.repeat(n, 1)
        # nodevec2 = (nodevec2.T.mm(self.w2) + self.b2.repeat(n, 1)).T
        # self.graph2 = F.leaky_relu_(torch.mm(nodevec1, nodevec2))
        # graph.append(self.graph2)
        inputs = torch.cat((inputs,torch.zeros(inputs.shape).to(self.device)),dim=-1)
        inputs, states = self.encoder(inputs, graph)
        outputs = self.decoder(inputs, graph, states)
        return outputs, graph

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))


class DynamicGraphNet(nn.Module):
    def __init__(self,
                 eigen_dim_origin,
                 eigen_dim_time,
                 hidden_dim,
                 input_dim,
                 horizon
                 ):
        super(DynamicGraphNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = eigen_dim_origin + eigen_dim_time
        self.input_dim = input_dim
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(self.emb_dim + self.input_dim * self.horizon, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.emb_dim))

    def set_require_gard(self, require = False):
        if(require == False):
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, inputs: Tensor):
        outputs = self.net(inputs)  # [batch, num_sensors, emb_dim]
        return outputs
