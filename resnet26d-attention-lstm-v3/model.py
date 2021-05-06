from common import *
from configure import *
from pretrain_model.resnet_26d import *
from torch.nn.utils.rnn import pack_padded_sequence


# https://arxiv.org/pdf/1411.4555.pdf
# 'Show and Tell: A Neural Image Caption Generator' - Oriol Vinyals, cvpr-2015


image_dim   = 2048
text_dim    = 256
decoder_dim = 512
attention_dim = 256



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

class Attention(nn.Module):
    def __init__(self):

        super(Attention, self).__init__()
        self.encoder_project = nn.Linear(image_dim, attention_dim)
        self.decoder_project = nn.Linear(decoder_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.gate    = nn.Linear(decoder_dim, image_dim)  # linear layer to create a sigmoid-activated gate


    def forward(self, image_embed, decoder_hidden):
        batch_size, num_pixel, c  = image_embed.shape

        x1 = self.encoder_project(image_embed   )               # (batch_size, num_pixel, attention_dim)
        x2 = self.decoder_project(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        x = x1 + x2

        a = self.attention(F.relu(x))   # (batch_size, num_pixel, attention_dim)
        weight = self.softmax(a)
        weighted_image_embed = (image_embed * weight).sum(dim=1)  # (batch_size, image_dim)

        gate = torch.sigmoid(self.gate(decoder_hidden))
        weighted_image_embed = gate * weighted_image_embed

        weight = weight.reshape(batch_size, num_pixel)
        return weighted_image_embed, weight


class Net(nn.Module):
    def init_hidden_state(self, image_embed):
        m = image_embed.mean(dim=1)
        h = self.init_h(m)  # (batch_size, decoder_dim)
        c = self.init_c(m)
        return h, c

    def __init__(self,):
        super(Net, self).__init__()
        self.encoder = Encoder()

        #---
        self.init_h = nn.Linear(image_dim, decoder_dim)
        self.init_c = nn.Linear(image_dim, decoder_dim)
        self.attention = Attention()

        self.embed = nn.Embedding(vocab_size, text_dim)
        self.logit = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        self.rnn = nn.LSTMCell(image_dim + text_dim, decoder_dim, bias=True)


        #----
        # initialization
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)


    def forward(self, image, token, length):

        image_embed = self.encoder(image)
        batch_size,c,h,w  = image_embed.shape
        num_pixel = w*h

        image_embed = image_embed.permute(0, 2, 3, 1).contiguous()
        image_embed = image_embed.reshape(batch_size,num_pixel, image_dim)

        text_embed = self.embed(token)
        h, c = self.init_hidden_state(image_embed)

        decode_length = [l-1 for l in length]
        max_decode_length = max(decode_length)
        logit  = torch.zeros(batch_size, max_length, vocab_size).to(image_embed.device)
        weight = torch.zeros(batch_size, max_length, num_pixel).to(image_embed.device)

        for t in range(max_decode_length):
            B = sum([l > t for l in decode_length])

            weighted_image_embed, w = self.attention(image_embed[:B], h[:B])

            h, c = self.rnn(
                torch.cat([text_embed[:B, t, :], weighted_image_embed], dim=1),
                (h[:B], c[:B])
            )  # (B, decoder_dim)

            l = self.logit(self.dropout(h))  # (batch_size_t, vocab_size)
            logit [:B, t, :] = l
            weight[:B, t, :] = w

            #<todo> forced teacher training?
            zz=0
        return logit



    def forward_beam_search_decode(self, image):
        #<todo> : beam_search decode
        return 0

    #@torch.jit.export()
    def forward_argmax_decode(self, image):
        batch_size = len(image)
        device = image.device

        image_embed = self.encoder(image)
        image_embed = image_embed.permute(0, 2, 3, 1).contiguous()
        image_embed = image_embed.reshape(batch_size, num_pixel, image_dim)

        # start token for LSTM input
        token = torch.full((batch_size,), fill_value=STOI['<sos>'], dtype=torch.long, device=device)
        h, c = self.init_hidden_state(image_embed)  # (batch_size, decoder_dim)

        #-----
        eos = torch.LongTensor([STOI['<eos>']]).to(device)
        pad = torch.LongTensor([STOI['<pad>']]).to(device)

        probability = torch.zeros(batch_size, max_length, vocab_size, device=device)
        predict = torch.full((batch_size, max_length), fill_value=STOI['<pad>'], dtype=torch.long, device=device)
        for t in range(max_length):
            text_embed = self.embed(token)
            weighted_image_embed, w = self.attention(image_embed, h)

            h, c = self.rnn(torch.cat([text_embed, weighted_image_embed], dim=1), (h, c))
            l = self.logit(h)
            p = F.softmax(l,-1)  # (1, vocab_size)
            k = torch.argmax(l, -1) #predict max

            probability[:, t, :] = p
            predict[:, t] = k
            token = k #next token
            if ((k == eos) | (k == pad)).all():  break

        return predict, probability



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