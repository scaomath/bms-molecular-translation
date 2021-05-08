from common import *
from bms import *
from configure import *

#from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def pad_sequence_to_max_length(sequence, max_length, padding_value):
    batch_size =len(sequence)
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L, ...] = s
    return pad_sequence

def load_tokenizer():
    tokenizer = YNakamaTokenizer(is_load=True)
    print('len(tokenizer) : vocab_size', len(tokenizer))
    for k,v in STOI.items():
        assert  tokenizer.stoi[k]==v
    return tokenizer


#(2424186, 6)
#Index(['image_id', 'InChI', 'formula', 'text', 'sequence', 'length'], dtype='object')
def make_fold(mode='train-1'):
    if 'train' in mode:
        df = read_pickle_from_file(data_dir+'/df_train.more.csv.pickle')
        df_fold = pd.read_csv(data_dir+'/df_fold.csv')
        df = df.merge(df_fold, on='image_id')
        df.loc[:,'path']='train'
        df.loc[:, 'orientation'] = 0

        df['fold'] = df['fold'].astype(int)
        #print(df.groupby(['fold']).size()) #404_031
        #print(df.columns)

        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_train, df_valid

    # Index(['image_id', 'InChI'], dtype='object')
    if 'test' in mode:
        df = pd.read_csv(data_dir+'/sample_submission.csv')
        df_orientation = pd.read_csv(data_dir+'/test_orientation.csv')
        df = df.merge(df_orientation, on='image_id')

        df.loc[:, 'path'] = 'test'
        df.loc[:, 'InChI'] = '0'
        df.loc[:, 'formula'] = '0'
        df.loc[:, 'text'] =  '0'
        df.loc[:, 'sequence'] = pd.Series([[0]] * len(df))
        df.loc[:, 'length'] = 1

        df_test = df
        return df_test


#make_fold(mode='test')
#####################################################################################################
class FixNumSampler(Sampler):
    def __init__(self, dataset, length=-1, is_shuffle=False):
        if length<=0:
            length=len(dataset)

        self.is_shuffle = is_shuffle
        self.length = length


    def __iter__(self):
        index = np.arange(self.length)
        if self.is_shuffle: random.shuffle(index)
        return iter(index)

    def __len__(self):
        return self.length



# see https://www.kaggle.com/yasufuminakama/inchi-resnet-lstm-with-attention-inference/data
def remote_augment(r):
    image = r['image']
    h, w, _ = image.shape

    # if h > w:
    #     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    l= r['d'].orientation
    if l == 1:
        image = np.rot90(image, -1)
    if l == 2:
        image = np.rot90(image, 1)
    if l == 3:
        image = np.rot90(image, 2)

    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    r['image'] = image
    return r

def null_augment(r):
    image = r['image']
    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    r['image'] = image
    return r

class BmsDataset(Dataset):
    def __init__(self, df, tokenizer, augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]

        image_file = data_dir +'/%s/%s/%s/%s/%s.png'%(d.path, d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
        image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        token = d.sequence
        r = {
            'index' : index,
            'image_id' : d.image_id,
            'InChI' : d.InChI,
            'formula' : d.formula,
            'd' : d,
            'image' : image,
            'token' : token,
        }
        if self.augment is not None: r = self.augment(r)
        return r


def null_collate(batch, is_sort_decreasing_length=True):
    collate = defaultdict(list)

    if is_sort_decreasing_length: #sort by decreasing length
        sort  = np.argsort([-len(r['token']) for r in batch])
        batch = [batch[s] for s in sort]

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)
    #----

    collate['length'] = [len(l) for l in collate['token']]

    token  = [np.array(t,np.int32) for t in collate['token']]
    token  = pad_sequence_to_max_length(token, max_length=max_length, padding_value=STOI['<pad>'])
    collate['token'] = torch.from_numpy(token).long()

    image = np.stack(collate['image'])
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image).unsqueeze(1).repeat(1,3,1,1)

    return collate



##############################################################################################################




def run_check_dataset():
    tokenizer = load_tokenizer()
    df_train, df_valid = make_fold('train-1')

    # df_train = make_fold('test') #1616107
    # dataset = BmsDataset(df_train, tokenizer, remote_augment)

    dataset = BmsDataset(df_train, tokenizer)
    print(dataset)


    # for i in range(len(dataset)):
    for i in range(100):
        #i = np.random.choice(len(dataset))
        r = dataset[i]

        print(r['index'])
        print(r['image_id'])
        print(r['formula'])
        print(r['InChI'])
        print(r['token'])

        print('image : ')
        print('\t', r['image'].shape)
        print('')

        #---

        image_show('image',r['image'], resize=1)
        cv2.waitKey(0)
        #exit(0)

    #exit(0)
    loader = DataLoader(
        dataset,
        sampler = RandomSampler(dataset),
        batch_size  = 8,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate,
    )
    for t,batch in enumerate(loader):
        if t>30: break

        print(t, '-----------')
        print('index : ', batch['index'])
        print('image : ')
        print('\t', batch['image'].shape, batch['image'].is_contiguous())
        print('length  : ')
        print('\t',len( batch['length']))
        print('\t', batch['length'])
        print('token  : ')
        print('\t', batch['token'].shape, batch['token'].is_contiguous())
        print('\t', batch['token'])

        print('')


# main #################################################################
if __name__ == '__main__':
    run_check_dataset()
    #run_check_augment()