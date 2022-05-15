import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents

        self.fc1 = nn.ModuleList([nn.Linear(input_shape, args.rnn_hidden_dim) for _ in range(self.n_agents)])
        self.rnn = nn.ModuleList([nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim) for _ in range(self.n_agents)])
        self.fc2 = nn.ModuleList([nn.Linear(args.rnn_hidden_dim, args.n_actions) for _ in range(self.n_agents)])

        self.noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.noise_embedding_dim)
        self.noise_fc2 = nn.Linear(args.noise_embedding_dim, args.noise_embedding_dim)
        self.noise_fc3 = nn.Linear(args.noise_embedding_dim, args.n_actions)

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, noise):
        agent_ids = th.eye(self.args.n_agents, device=inputs.device).repeat(noise.shape[0], 1)
        noise_repeated = noise.repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)

        bs = inputs.view(-1, self.n_agents, self.input_shape).shape[0]
        agent_inputs = [inputs.view(bs, self.n_agents, -1)[:, i, :] for i in range(self.n_agents)]
        hiddens = [hidden_state.view(bs, self.n_agents, -1)[:, i, :] for i in range(self.n_agents)]
        xs = [F.relu(fc1(ip)) for ip, fc1 in zip(agent_inputs, self.fc1)]
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = [rnn(x, h_in) for x, h_in, rnn in zip(xs, hiddens, self.rnn)]
        q = [fc2(h) for h, fc2 in zip(hs, self.fc2)]

        h = th.stack(hs, 1).view(bs * self.n_agents, -1)
        q = th.stack(q, 1).view(bs * self.n_agents, -1)

        noise_input = th.cat([noise_repeated, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_noise_fc1(noise_input).reshape(-1, self.args.n_actions, self.args.rnn_hidden_dim)
            wq = th.bmm(W, h.unsqueeze(2))
        else:
            z = F.tanh(self.noise_fc1(noise_input))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)

            wq = q * wz

        return wq, h
