from common import *
from configure import *
from tnt import *
from torch.nn.utils.rnn import pack_padded_sequence

from fairseq_transformer import *


# https://arxiv.org/pdf/1411.4555.pdf
# 'Show and Tell: A Neural Image Caption Generator' - Oriol Vinyals, cvpr-2015

class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()

        e = tnt_s_patch16_224(pretrained=True)
        self.e = e

    def forward(self, image):
        batch_size, C, H, W = image.shape
        x = 2 * image - 1  # ; print('input ',   x.size())


        pixel_embed = self.e.pixel_embed(x, self.e.pixel_pos)

        patch_embed = self.e.norm2_proj(self.e.proj(self.e.norm1_proj(pixel_embed.reshape(batch_size, self.e.num_patches, -1))))
        patch_embed = torch.cat((self.e.cls_token.expand(batch_size, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.e.patch_pos
        patch_embed = self.e.pos_drop(patch_embed)

        for blk in self.e.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        patch_embed = self.e.norm(patch_embed) #torch.Size([7, 197, 384])


        x = patch_embed
        return x
'''
224/16 = 14
14*14 = 196


'''

image_dim   = 384
text_dim    = 384
decoder_dim = 384
num_layer = 3
num_head  = 8
ff_dim = 1024


class Net(nn.Module):

    def __init__(self,):
        super(Net, self).__init__()
        self.cnn = CNN()
        self.image_encode = nn.Identity()

        #---
        self.text_pos    = PositionEncode1D(text_dim,max_length)
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        #---
        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        #----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)


    @torch.jit.unused
    def forward(self, image, token, length):
        device = image.device
        batch_size = len(image)
        #---

        image_embed = self.cnn(image)
        image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()

        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous()

        text_mask = np.triu(np.ones((max_length, max_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)

        #----
        # <todo> mask based on length of token?
        # <todo> perturb mask as aug

        x = self.text_decode(text_embed, image_embed, text_mask)
        x = x.permute(1,0,2).contiguous()

        logit = self.logit(x)
        return logit


    @torch.jit.export
    def forward_argmax_decode(self, image):

        image_dim = 384
        text_dim = 384
        decoder_dim = 384
        num_layer = 3
        num_head = 8
        ff_dim = 1024

        STOI = {
            '<sos>': 190,
            '<eos>': 191,
            '<pad>': 192,
        }

        image_size = 224
        vocab_size = 193
        max_length = 300  # 275


        #---------------------------------
        device = image.device
        batch_size = len(image)

        image_embed = self.cnn(image)
        image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()

        token = torch.full((batch_size, max_length), STOI['<pad>'],dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:,0] = STOI['<sos>']


        #-------------------------------------
        eos = STOI['<eos>']
        pad = STOI['<pad>']
        # https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/tests/test_consistency.py
        # slow version
        # if 0:
        #     for t in range(max_length-1):
        #         last_token = token [:,:(t+1)]
        #         text_embed = self.token_embed(last_token)
        #         text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous() #text_embed + text_pos[:,:(t+1)] #
        #
        #         text_mask = np.triu(np.ones((t+1, t+1)), k=1).astype(np.uint8)
        #         text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)
        #
        #         x = self.text_decode(text_embed, image_embed, text_mask)
        #         x = x.permute(1,0,2).contiguous()
        #
        #         l = self.logit(x[:,-1])
        #         k = torch.argmax(l, -1)  # predict max
        #         token[:, t+1] = k
        #         if ((k == eos) | (k == pad)).all():  break

        # fast version
        if 1:
            #incremental_state = {}
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
            )
            for t in range(max_length-1):
                #last_token = token [:,:(t+1)]
                #text_embed = self.token_embed(last_token)
                #text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

                last_token = token[:, t]
                text_embed = self.token_embed(last_token)
                text_embed = text_embed + text_pos[:,t] #
                text_embed = text_embed.reshape(1,batch_size,text_dim)

                x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
                x = x.reshape(batch_size,decoder_dim)
                #print(incremental_state.keys())

                l = self.logit(x)
                k = torch.argmax(l, -1)  # predict max
                token[:, t+1] = k
                if ((k == eos) | (k == pad)).all():  break

        predict = token[:, 1:]
        return predict




# loss #################################################################
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss

# https://www.aclweb.org/anthology/2020.findings-emnlp.276.pdf
def seq_anti_focal_cross_entropy_loss(logit, token, length):
    gamma = 0.5 # {0.5,1.0}
    label_smooth = 0.90

    #---
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    #loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    #non_pad = torch.where(truth != STOI['<pad>'])[0]  # & (t!=STOI['<sos>'])


    # ---
    #p = F.softmax(logit,-1)
    #logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))

    logp = F.log_softmax(logit, -1)
    logp = logp.gather(1, truth.reshape(-1,1)).reshape(-1)
    p = logp.exp()

    #loss = - ((1 - p) ** gamma)*logp  #focal
    loss = - ((1 + p) ** gamma)*logp  #anti-focal
    loss = loss.mean()
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
    loss = seq_anti_focal_cross_entropy_loss(logit, token, length)


    print('vocab_size',vocab_size)
    print('max_length',max_length)
    print('')
    print(length)
    print(length.shape)
    print(token.shape)
    print(image.shape)
    print('---')

    print(logit.shape)
    print(loss)
    print('---')


    #---
    print('torch.jit.script(net)')
    net.eval()
    net = torch.jit.script(net)

    predict = net.forward_argmax_decode(image)
    print(predict.shape)


# main #################################################################
if __name__ == '__main__':
     run_check_net()