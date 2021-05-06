from common import *
from configure import *
from pretrain_model.resnet_26d import *
from torch.nn.utils.rnn import pack_padded_sequence


# https://arxiv.org/pdf/1411.4555.pdf
# 'Show and Tell: A Neural Image Caption Generator' - Oriol Vinyals, cvpr-2015


image_dim   = 512
text_dim    = 512
decoder_dim = 1024


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.sigmoid(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        return grad_output * (sigmoid * (1 + x * (1 - sigmoid)))
F_swish = SwishFunction.apply

class Swish(nn.Module):
    def forward(self, x):
        return F_swish(x)


class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()

        e = make_resnet_26d()
        pretrain_state_dict = torch.load(PRETRAIN_CHECKPOINT, map_location=lambda storage, loc: storage)
        print(e.load_state_dict(pretrain_state_dict, strict=True))

        # ---
        self.p1 = nn.Sequential(
            e.conv1,
            e.bn1,
            e.act1,
        )
        self.p2 = nn.Sequential(
            e.maxpool,
            e.layer1,
        )
        self.p3 = e.layer2
        self.p4 = e.layer3
        self.p5 = e.layer4
        del e  # dropped

    def forward(self, image):
        batch_size, C, H, W = image.shape
        x = 2 * image - 1  # ; print('input ',   x.size())

        x = self.p1(x)  # ; print('c1 ',c1.size())
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        return x


class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.image_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(2048,image_dim),
            nn.BatchNorm1d(image_dim),
            Swish()
        )
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.logit = nn.Linear(decoder_dim, vocab_size)

        self.rnn = nn.LSTM(
            text_dim,
            decoder_dim,
            num_layers = 2,
            bias = True,
            batch_first = True,
            dropout = 0.2,
            bidirectional = False
        )
        #----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)


    def forward(self, image, token, length):
        batch_size,c,h,w  = image.shape

        image_embed = self.encoder(image)
        image_embed = self.image_embed(image_embed)
        text_embed  = self.token_embed(token)

        x = torch.cat([image_embed.unsqueeze(1),text_embed],1)
        y , (h,c) = self.rnn(x)
        logit = self.logit(y[:,1:])
        return logit


    def predict(self, image):
        #<todo> : greedy argmax decode
        return 0

# loss #################################################################
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss



# check #################################################################
def run_check_net():
    batch_size = 7
    C,H,W = 3, 224, 224
    image = torch.randn((batch_size,C,H,W))

    token  = np.full((batch_size, max_length), STOI['<pad>'], np.int64) #token
    length = np.random.randint(5,max_length-2, batch_size)
    length = np.sort(length)[::-1].copy()
    for b in range(batch_size):
        l = length[b]
        t = np.random.choice(vocab_size,l)
        t = np.insert(t,0,     STOI['<sos>'])
        t = np.insert(t,len(t),STOI['<eos>'])
        L = len(t)
        token[b,:L]=t

    token  = torch.from_numpy(token).long()



    #---
    net = Net()
    net.train()

    logit = net(image, token, length)
    print('vocab_size',vocab_size)
    print('max_length',max_length)
    print('')
    print(length)
    print(length.shape)
    print(token.shape)
    print(image.shape)
    print('---')

    print(logit.shape)
    print('---')



# main #################################################################
if __name__ == '__main__':
     run_check_net()