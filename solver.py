import torch.nn as nn

from rl_with_attention import AttentionTSP

from utils import get_solution_reward


class Solver(nn.Module):
    def __init__(self):
        super(Solver, self).__init__()

    def reward(self, sample_solution):
        """
        [Input]
            sample_solution: batch x seq_len x feature_size -> torch.LongTensor
        [Return]
            tour_len: batch
        """

        tour_len = get_solution_reward(sample_solution)

        return tour_len

    def forward(self, inputs):
        """
        [Input]
            inputs: batch x seq_len x feature
        [Return]
            R:
            probs:
            actions: batch x seq_len
        """

        probs, actions = self.actor(inputs)

        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, inputs.shape[-1])))

        return R, probs, actions

class solver_Attention(Solver):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_Attention, self).__init__()

        self.actor = AttentionTSP(embedding_size,
                                  hidden_size,
                                  seq_len)
