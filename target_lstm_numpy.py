import numpy as np

def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
def logsumexp(x, axis=None, keepdims=False):
    max = np.max(x, axis=axis, keepdims=keepdims)
    rx = x - max
    sumexp = np.sum(np.exp(rx), axis=axis, keepdims=keepdims)
    return max + np.log(sumexp)
def log_softmax(x):
    return x - logsumexp(x, axis=-1, keepdims=True)
def zipmean(logits, idx):
    assert np.shape(logits)[0] == np.shape(idx)[0]
    ret = 0.0
    for i in range(logits.shape[0]):
        ret += logits[i, idx[i]]
    return ret / logits.shape[0]

def multinomial(x):
    ret = []
    for i in range(x.shape[0]):
        p_val = x[i]
        ret.append(np.random.multinomial(1, p_val))
    return np.stack(ret)


class TARGET_LSTM(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, params):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.g_params = []
        self.temperature = 1.0
        self.params = params

        np.random.seed(66)
        self.g_embeddings = self.params[0]
        self.g_params.append(self.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        self.h0 = np.zeros([self.batch_size, self.hidden_dim])


    def generate(self, batch_size=64):
        # h0 = np.random.normal(size=self.hidden_dim)
        h, c = np.zeros([batch_size, self.hidden_dim]), np.zeros([batch_size, self.hidden_dim])
        x_t = np.array([self.start_token] * batch_size, dtype=np.int32)
        gen_x = []
        for i in range(self.sequence_length):
            emb_x_t = self.g_embeddings[x_t]
            h, c = self.g_recurrent_unit(emb_x_t, (h, c))  # hidden_memory_tuple
            o_t = self.g_output_unit((h, c))  # batch x vocab , logits not prob
            prob = softmax(o_t)
            x_t = multinomial(prob).argmax(axis=-1)
            gen_x.append(x_t)
        return np.stack(gen_x).transpose([1, 0])

    def calc_nll(self, x):
        batch_size = x.shape[0]
        h, c = np.zeros([batch_size, self.hidden_dim]), np.zeros([batch_size, self.hidden_dim])
        x_t = np.array([self.start_token] * batch_size, dtype=np.int32)
        feed_x = x.transpose([1, 0])
        nll = 0.0
        for i in range(self.sequence_length):
            emb_x_t = self.g_embeddings[x_t]
            x_t = feed_x[i]
            h, c = self.g_recurrent_unit(emb_x_t, (h, c))  # hidden_memory_tuple
            o_t = self.g_output_unit((h, c))  # batch x vocab , logits not prob
            prob = log_softmax(o_t)
            nll += zipmean(prob, x_t)
        return -nll / self.sequence_length
    def init_matrix(self, shape):
        return np.random.normal(shape, scale=1.0)

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = self.params[1]
        self.Ui = self.params[2]
        self.bi = self.params[3]

        self.Wf = self.params[4]
        self.Uf = self.params[5]
        self.bf = self.params[6]

        self.Wog = self.params[7]
        self.Uog = self.params[8]
        self.bog = self.params[9]

        self.Wc = self.params[10]
        self.Uc = self.params[11]
        self.bc = self.params[12]

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = hidden_memory_tm1

            # Input Gate

            i = sigmoid(
                x @ self.Wi + previous_hidden_state @ self.Ui + self.bi
            )

            # Forget Gate
            f = sigmoid(
                 x @ self.Wf + previous_hidden_state @ self.Uf + self.bf
            )

            # Output Gate
            o = sigmoid(
                x @ self.Wog + previous_hidden_state @ self.Uog + self.bog
            )

            # New Memory Cell
            c_ = np.tanh(
                x @ self.Wc + previous_hidden_state @ self.Uc + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * np.tanh(c)

            return (current_hidden_state, c)

        return unit

    def create_output_unit(self):
        self.Wo = self.params[13]
        self.bo = self.params[14]

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = hidden_memory_tuple
            # hidden_state : batch x hidden_dim
            logits = hidden_state @ self.Wo + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit
