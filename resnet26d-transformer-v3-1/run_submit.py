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
        '/root/share1/kaggle/2021/bms-moleular-translation/result/try10/resnet26d-224-transformer/fold%d-1' % fold
    initial_checkpoint = \
        out_dir + '/checkpoint/00146000_model.pth'#


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
            #df_valid = df_valid[:40_000]
            df_valid = df_valid[:5000]

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

len(tokenizer) : vocab_size 193
<All keys matched successfully>
     5000 / 5000   9 min 46 sec
submit_dir : /root/share1/kaggle/2021/bms-moleular-translation/result/try10/resnet26d-224-transformer/fold3-1/valid/local-00098000_model1
initial_checkpoint : /root/share1/kaggle/2021/bms-moleular-translation/result/try10/resnet26d-224-transformer/fold3-1/checkpoint/00098000_model.pth
df_submit : (5000, 2)
          image_id                                              InChI
0     00006b137aef  InChI=1S/C22H25NO5S/c1-29(26,27)20-14-12-17(13...
1     000077bfb356  InChI=1S/C16H23NO3/c1-4-17(5-2)14(18)11-12(3)1...
2     000093a52a28  InChI=1S/C13H19N3O4/c17-12(18)6-16-11(14-13(15...
3     0000a651dd85  InChI=1S/C29H24N4O4S/c1-36-23-10-7-20(31-26(35...
4     0000d93e0e9b  InChI=1S/C31H40FNO4/c1-30-15-13-23(34)19-21(30...
...            ...                                                ...
4995  032f3f7ede40  InChI=1S/C20H31N5O2/c1-3-21-20(27)25-15-13-24(...
4996  032f53eb25b6  InChI=1S/C11H13Cl2N/c1-7-4-9(12)10(13)5-8(7)11...
4997  032f57610c1e  InChI=1S/C12H15N3O3S/c1-9(10-4-5-10)15-8-11(7-...
4998  032f83a3203a  InChI=1S/C7H12O3/c1-5(8)6-3-2-4-7(9)10-6/h6-7,...
4999  033055283223  InChI=1S/C20H16N4O/c25-20(22-16-5-3-4-14-12-21...

[5000 rows x 2 columns]

lb_score  = 5.211800
is_norm_ichi = False

-------------------

<All keys matched successfully>
     5000 / 5000   3 min 55 sec
submit_dir : /root/share1/kaggle/2021/bms-moleular-translation/result/try10/resnet26d-224-transformer/fold3-1/valid/local-00098000_model1
initial_checkpoint : /root/share1/kaggle/2021/bms-moleular-translation/result/try10/resnet26d-224-transformer/fold3-1/checkpoint/00098000_model.pth
df_submit : (5000, 2)
          image_id                                              InChI
0     00006b137aef  InChI=1S/C22H25NO5S/c1-29(26,27)20-14-12-17(13...
1     000077bfb356  InChI=1S/C16H23NO3/c1-4-17(5-2)14(18)11-12(3)1...
2     000093a52a28  InChI=1S/C13H19N3O4/c17-12(18)6-16-11(14-13(15...
3     0000a651dd85  InChI=1S/C29H24N4O4S/c1-36-23-10-7-20(31-26(35...
4     0000d93e0e9b  InChI=1S/C31H40FNO4/c1-30-15-13-23(34)19-21(30...
...            ...                                                ...
4995  032f3f7ede40  InChI=1S/C20H31N5O2/c1-3-21-20(27)25-15-13-24(...
4996  032f53eb25b6  InChI=1S/C11H13Cl2N/c1-7-4-9(12)10(13)5-8(7)11...
4997  032f57610c1e  InChI=1S/C12H15N3O3S/c1-9(10-4-5-10)15-8-11(7-...
4998  032f83a3203a  InChI=1S/C7H12O3/c1-5(8)6-3-2-4-7(9)10-6/h6-7,...
4999  033055283223  InChI=1S/C20H16N4O/c25-20(22-16-5-3-4-14-12-21...

[5000 rows x 2 columns]

lb_score  = 5.214000
is_norm_ichi = False


--------
[10000 rows x 2 columns]

lb_score  = 5.476700
is_norm_ichi = False



--------
[40000 rows x 2 columns]

lb_score  = 5.405725
is_norm_ichi = False

----------
try10/resnet26d-224-transformer/fold3-1/checkpoint/00110000_model.pth
[5000 rows x 2 columns]

lb_score  = 3.647400
is_norm_ichi = False


[40000 rows x 2 columns]

lb_score  = 3.705700
is_norm_ichi = False


try10/resnet26d-224-transformer/fold3-1/checkpoint/00128000_model.pth
[5000 rows x 2 columns]

lb_score  = 3.690800
is_norm_ichi = False




initial_checkpoint : /root/share1/kaggle/2021/bms-moleular-translation/result/try10/resnet26d-224-transformer/fold3-1/checkpoint/00146000_model.pth
[5000 rows x 2 columns]

lb_score  = 3.469800
is_norm_ichi = False

'''
