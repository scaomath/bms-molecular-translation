from common import *
from bms import *

from fairseq_model import *
#---------------------------------------


def run_check_fairseq():
    net = Net()
    net.eval()

    if 1:
        fold = 3
        out_dir = \
            '/root/share1/kaggle/2021/bms-moleular-translation/result/try10/tnt-s-224-fairseq/fold%d' % fold
        initial_checkpoint = \
            out_dir + '/checkpoint/00235000_model.pth'#

        #---------------------------------------------------------------------------------
        state_dict = torch.load(initial_checkpoint)['state_dict']
        net.load_state_dict(state_dict, strict=True)

        def load_tokenizer():
            tokenizer = YNakamaTokenizer(is_load=True)
            print('len(tokenizer) : vocab_size', len(tokenizer))
            for k, v in STOI.items():
                assert tokenizer.stoi[k] == v
            return tokenizer

        tokenizer = load_tokenizer()
        data_dir = '/root/share1/kaggle/2021/bms-moleular-translation/data'
        #df_train = pd.read_csv(data_dir + '/train_labels.csv')
        df_train = read_pickle_from_file(data_dir+'/df_train.more.csv.pickle')


        batch_size = 8
        token = np.full((batch_size,max_length),STOI['<pad>'])
        image=[]
        length=[]
        for i,d in df_train.iterrows():
            if i==batch_size: break
            image_file = data_dir + '/%s/%s/%s/%s/%s.png' % ('train', d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
            m = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
            image.append(m)

            L= len(d.sequence)
            token[i,:L]=d.sequence
            length.append(L)


        image = np.stack(image).astype(np.float32)/255
        image = torch.from_numpy(image).unsqueeze(1).repeat(1,3,1,1)


        token = torch.from_numpy(token).long()
        text = df_train.InChI.values[:batch_size]




        net.eval()
        net.cuda()
        image = image.cuda()
        token = token.cuda()

        start_timer = timer()

        if 0: #train
            k = net.forward(image, token, length)
            k = k.argmax(-1)

        if 1:#incremental
            k = net.forward_argmax_decode(image)

        print( time_to_str(timer() - start_timer, 'sec'))
        k = k.data.cpu().numpy()
        for i in range(len(k)):
            p = tokenizer.one_predict_to_inchi(k[i])
            print(p)
            print(text[i])
            print('')



# main #################################################################
if __name__ == '__main__':
    run_check_fairseq()
    pass