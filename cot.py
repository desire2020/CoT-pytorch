import torch as tc
import torch.cuda as tcg
import torch.autograd as train
import torch.utils.data as tcdata
import tqdm
import pickle
from target_lstm_numpy import TARGET_LSTM
from generator import Generator

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 0 # supervise (maximum likelihood estimation) epochs (not recommended)
SEED = 88
BATCH_SIZE = 64
VOCAB_SIZE = 5000
M_DROPOUT_RATE = 0.5 # Dropout rate of M (optional)
RESTORE = False
GRAD_ANALYSIS = False

def main():
    target_params = pickle.load(open('save/target_params_py3.pkl', 'rb'))
    target_lstm = TARGET_LSTM(VOCAB_SIZE, BATCH_SIZE, 32, 32, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model
    train_data = target_lstm.generate(batch_size=10000)
    generator = Generator(VOCAB_SIZE, BATCH_SIZE, 32, 32, SEQ_LENGTH, START_TOKEN, learning_rate=1e-3)
    mediator = Generator(VOCAB_SIZE, BATCH_SIZE, 64, 64, SEQ_LENGTH, START_TOKEN, learning_rate=1e-3)
    data_loader = tcdata.DataLoader(
        tcdata.TensorDataset(tcg.LongTensor(train_data)),
        batch_size=32, shuffle=True
    )
    log_cot = open("save/cot.log", "w")
    for epoch in range(20000):
        for i, (x, ) in enumerate(data_loader):
            m_loss, _ = mediator.teacher_forcing(tc.cat((generator.generate(32, keep_torch=True), x), dim=0), is_train=True)
            gen_x = generator.generate(64)
            _, log_pred = mediator.teacher_forcing(gen_x)
            generator.cooperative_training(gen_x, log_pred)
            if i % 20 == 0:
                print("mediator loss at iteration #%d-%d" % (epoch, i), m_loss)
        print("oracle loss at epoch #%d" % epoch, target_lstm.calc_nll(generator.generate(64)))
        print("test loss at epoch #%d" % epoch, generator.teacher_forcing(target_lstm.generate(64))[0])
        print("oracle loss at epoch #%d" % epoch, target_lstm.calc_nll(generator.generate(64)), file=log_cot)
        print("test loss at epoch #%d" % epoch, generator.teacher_forcing(target_lstm.generate(64))[0], file=log_cot)
    log_cot.close()
if __name__ == "__main__":
    main()