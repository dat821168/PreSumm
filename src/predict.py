import os
import torch
import bisect
import argparse
import stanza

from pytorch_transformers import BertTokenizer
from src.others.logging import logger, init_logger
from src.models.model_builder import AbsSummarizer
from src.models.predictor import build_predictor
from src.models.data_loader import Batch
from src.prepro.data_builder import BertData

stanza.download('en')

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Translator(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.predictor = None
        self.load_checkpoint(self.args.test_from)
        self.bert_processor = BertData(self.args)
        self.tokenizer = stanza.Pipeline(lang='en', processors='tokenize')

    def load_checkpoint(self, test_from):
        logger.info('Loading checkpoint from %s' % test_from)
        checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(self.args, k, opt[k])
        logger.info(self.args)
        model = AbsSummarizer(self.args, self.device, checkpoint)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=self.args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
        self.predictor = build_predictor(self.args, tokenizer, symbols, model, logger)

    def preprocess(self, src):
        doc = self.tokenizer(src)
        src_sents = []
        for i, sentence in enumerate(doc.sentences):
            src_sents.append([token.text.lower() for token in sentence.tokens])
        sent_labels = [0] * len(src_sents)
        return src_sents, sent_labels

    def postprocess(self, ex):
        src = ex[0]
        tgt = ex[2][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex[1]
        segs = ex[3]
        if not self.args.use_interval:
            segs = [0] * len(segs)
        clss = ex[4]
        src_txt = ex[5]
        tgt_txt = ex[6]

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt

    def build_batch(self, srcs):
        data = []
        for src in srcs:
            src_sents, sent_labels = self.preprocess(src)
            example = self.bert_processor.preprocess(src_sents, tgt=[""], sent_labels=sent_labels, is_test=True)
            data.append(self.postprocess(example))
        batch = Batch(data, self.device, is_test=True)
        return batch

    def translate(self, src):
        batch = self.build_batch(src)
        pred_str = self.predictor.single_translate(batch)
        return pred_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-data_path", default='../samples/sample_1.txt')
    parser.add_argument("-test_from", default='../models/bertsumextabs_cnndm_final_model/model_step_148000.pt')
    parser.add_argument("-model_path", default='../models/bertsumextabs_cnndm_final_model/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='./temp')

    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-test_batch_size", default=1, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    translator = Translator(args, device)
    with open(args.data_path, "r") as f:
        full_text = f.read()
        f.close()
    print(translator.translate([full_text])[0])
