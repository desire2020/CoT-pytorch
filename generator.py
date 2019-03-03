import torch as tc
import torch.nn as nn
import torch.autograd as train
import torch.optim as opt
import torch.nn.functional as func
import torch.cuda as tcg
import numpy as np
class Generator(nn.Module):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token=0,
                 learning_rate=1e-3, reward_gamma=0.95, name="generator", dropout_rate=0.5):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.sequence_length = sequence_length
        self.vocab_size = num_emb
        self.start_token = 0

        self.is_cuda = tcg.is_available()

        self.g_embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.g_recurrent_unit = nn.LSTMCell(self.emb_dim, self.hidden_dim)
        self.g_output_unit = nn.Linear(self.hidden_dim, self.vocab_size)

        for p in self.parameters():
            nn.init.normal(p, 0, 0.1)
        self.g_opt = opt.Adam(params=self.parameters(),
                              lr=learning_rate, betas=(0.9, 0.95),
                              amsgrad=False)
        if self.is_cuda:
            self.cuda()

    def __h0_getter(self, batch_size=1):
        h = train.Variable(tc.zeros(batch_size, self.hidden_dim))

        if self.is_cuda:
            return h.cuda()
        else:
            return h

    def forward(self, x, h, c):
        emb_x = self.g_embeddings(x)
        h, c = self.g_recurrent_unit(emb_x, (h, c))
        logits = self.g_output_unit(h)
        log_prob = func.log_softmax(logits, dim=-1)
        return log_prob, h, c

    def generate(self, batch_size=64, keep_torch=False):

        h = self.__h0_getter(batch_size)
        c = self.__h0_getter(batch_size)

        x_t = train.Variable(tc.LongTensor(np.array([self.start_token] * batch_size, dtype=np.int32)))

        gen_x  = tc.zeros(batch_size, self.sequence_length).type(tc.LongTensor)

        if self.is_cuda:
            gen_x = gen_x.cuda()
            x_t = x_t.cuda()

        for i in range(self.sequence_length):
            log_prob, h, c = self.forward(x_t, h, c)
            x_t = tc.multinomial(tc.exp(log_prob), 1).view(-1)
            gen_x[:, i] = x_t.data
        if keep_torch:
            return gen_x
        return gen_x.cpu().numpy()

    def teacher_forcing(self, feed_x, is_train=False):
        loss_fn = nn.NLLLoss()
        if type(feed_x) is np.ndarray:
            feed_x = tcg.LongTensor(feed_x)
        batch_size, seq_len = feed_x.size()
        x_t = train.Variable(tc.LongTensor(np.array([self.start_token] * batch_size, dtype=np.int32)))
        log_g_prediction = train.Variable(tc.zeros(batch_size, seq_len, self.vocab_size))
        feed_x = feed_x.permute(1, 0)  # seq_len x batch_size
        h = self.__h0_getter(batch_size)
        c = self.__h0_getter(batch_size)

        if self.is_cuda:
            x_t = x_t.cuda()
        loss = 0
        for i in range(seq_len):
            log_pred, h, c = self.forward(x_t, h, c)
            x_t = feed_x[i]
            loss += loss_fn(log_pred, x_t)
            log_g_prediction[:, i, :] = log_pred
        loss /= self.sequence_length
        if is_train:
            self.g_opt.zero_grad()
            loss.backward()
            self.g_opt.step()
        return loss.detach().cpu().numpy(), log_g_prediction

    def cooperative_training(self, feed_x, m_log_pred):
        _, log_g_prediction = self.teacher_forcing(feed_x)
        loss = tc.sum(
            func.softmax(log_g_prediction, dim=-1) * (log_g_prediction - m_log_pred))
        loss /= feed_x.shape[0] * feed_x.shape[1]
        self.g_opt.zero_grad()
        loss.backward()
        self.g_opt.step()
