import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from bms import *

from lib.net.lookahead import *
from lib.net.radam import *

from model import *
from dataset_224 import *


# ----------------
is_mixed_precision = True #False  #


###################################################################################################
import torch.cuda.amp as amp
if is_mixed_precision:
    class AmpNet(Net):
        @torch.cuda.amp.autocast()
        def forward(self, *args):
            #return super(AmpNet, self).forward(*args)
            return super(AmpNet, self).forward_argmax_decode(*args)
else:
    AmpNet = Net


# start here ! ###################################################################################


def do_predict(net, tokenizer, valid_loader):

    text = []

    start_timer = timer()
    valid_num = 0
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image = batch['image'].cuda()


        net.eval()
        with torch.no_grad():
            k = net(image)

            # token = batch['token'].cuda()
            # length = batch['length']
            # logit = data_parallel(net,(image, token, length))
            # k = logit.argmax(-1)

            k = k.data.cpu().numpy()
            k = tokenizer.predict_to_inchi(k)
            text.extend(k)

        valid_num += batch_size
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)

    assert(valid_num == len(valid_loader.dataset))
    print('')
    return text






def run_submit():
    fold = 3
    out_dir = \
        '/root/share1/kaggle/2021/bms-moleular-translation/result/try10/resnet101d-224-transformer/fold%d' % fold
    initial_checkpoint = \
        out_dir + '/checkpoint/00094000_model.pth'#

    is_norm_ichi = False #True
    if 1:

        ## setup  ----------------------------------------
        mode = 'local'
        #mode = 'remote'

        submit_dir = out_dir + '/valid/%s-%s1'%(mode, initial_checkpoint[-18:-4])
        os.makedirs(submit_dir, exist_ok=True)

        log = Logger()
        log.open(out_dir + '/log.submit.txt', mode='a')
        log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
        log.write('is_norm_ichi = %s\n' % is_norm_ichi)
        log.write('\n')

        #
        ## dataset ------------------------------------
        tokenizer = load_tokenizer()

        if 'remote' in mode: #1_616_107
           df_valid = make_fold('test')

        if 'local' in mode:  #484_837
            df_train, df_valid = make_fold('train-%d' % fold)
            df_valid = df_valid[:40_000]
            #df_valid = df_valid[:5000]

 
        valid_dataset = BmsDataset(df_valid, tokenizer, augment=remote_unrotate_augment)
        valid_loader  = DataLoader(
            valid_dataset,
            sampler = SequentialSampler(valid_dataset),
            batch_size  = 32, #32,
            drop_last   = False,
            num_workers = 4,
            pin_memory  = True,
            collate_fn  = lambda batch: null_collate(batch,False),
        )
        log.write('mode : %s\n'%(mode))
        log.write('valid_dataset : \n%s\n'%(valid_dataset))

        ## net ----------------------------------------
        if 1:
            tokenizer = load_tokenizer()
            net = AmpNet().cuda()

            net.load_state_dict(torch.load(initial_checkpoint)['state_dict'], strict=True)

            #---
            predict = do_predict(net, tokenizer, valid_loader)

            #np.save(submit_dir + '/probability.uint8.npy',probability)
            #write_pickle_to_file(submit_dir + '/predict.pickle', predict)
            #exit(0)
        else:
            pass
            #predict = read_pickle_from_file(submit_dir + '/predict.pickle')

        #----
        if is_norm_ichi:
            predict = [normalize_inchi(t) for t in predict]  #

        df_submit = pd.DataFrame()
        df_submit.loc[:,'image_id'] = df_valid.image_id.values
        df_submit.loc[:,'InChI'] = predict #
        df_submit.to_csv(submit_dir + '/submit.csv', index=False)

        log.write('submit_dir : %s\n' % (submit_dir))
        log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
        log.write('df_submit : %s\n' % str(df_submit.shape))
        log.write('%s\n' % str(df_submit))
        log.write('\n')

        if 'local' in mode:
            truth = df_valid['InChI'].values.tolist()
            lb_score = compute_lb_score(predict, truth)
            #print(lb_score)
            log.write('lb_score  = %f\n'%lb_score.mean())
            log.write('is_norm_ichi = %s\n' % is_norm_ichi)
            log.write('\n')

            if 1:
                df_eval = df_submit.copy()
                df_eval.loc[:,'truth']=truth
                df_eval.loc[:,'lb_score']=lb_score
                df_eval.loc[:,'length'] = df_valid['length']
                df_eval.to_csv(submit_dir + '/df_eval.csv', index=False)
                df_valid.to_csv(submit_dir + '/df_valid.csv', index=False)


        exit(0)

# main #################################################################
if __name__ == '__main__':
    run_submit()

'''
train 
0.00000  5.2000* 3.01  | 0.035  7.000  | 0.000  0.000  0.000  |  0 hr 00 min

submit (forward with mask)
lb_score  = 1.367600

submit (fast)
5000 / 5000   5 min 09 sec
lb_score  = 4.905200
is_norm_ichi = False


[5000 rows x 2 columns]

lb_score  = 4.625800
is_norm_ichi = False

---
00094000_model


lb_score  = 3.335800
is_norm_ichi = False


[40000 rows x 2 columns]

lb_score  = 3.377325
is_norm_ichi = False



'''
