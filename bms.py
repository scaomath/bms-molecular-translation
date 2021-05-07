from common import *
import Levenshtein

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

data_dir = '/media/scao/Data/bms_data'
#https://www.kaggle.com/nofreewill/normalize-your-predictions

def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            try:
                inchi = Chem.MolToInchi(mol)
            except:
                pass
    except:
        pass
    return inchi


# -----------------------------------------------------------------------

class YNakamaTokenizer(object):

    def __init__(self, is_load=True):
        self.stoi = {}
        self.itos = {}

        if is_load:
            self.stoi = read_pickle_from_file(data_dir+'/tokenizer.stoi.pickle')
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def build_vocab(self, text):
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {k: v for v, k in self.stoi.items()}

    def one_text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def one_sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def one_predict_to_inchi(self, predict):
        inchi = 'InChI=1S/'
        for p in predict:
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi

    # ---
    def text_to_sequence(self, text):
        sequence = [
            self.one_text_to_sequence(t)
            for t in text
        ]
        return sequence

    def sequence_to_text(self, sequence):
        text = [
            self.one_sequence_to_text(s)
            for s in sequence
        ]
        return text

    def predict_to_inchi(self, predict):
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi


# -----------------------------------------------------------------------



def compute_lb_score(predict, truth):
    score = []
    for p, t in zip(predict, truth):
        s = Levenshtein.distance(p, t)
        score.append(s)
    score = np.array(score)
    return score

