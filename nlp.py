
'''
Biaffine Dependency parser from AllenNLP
'''
import argparse
import json
import os
import re
import sys

from allennlp.predictors.predictor import Predictor
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

MODELS_DIR = '/data1/yangyy/pretrained-models'
model_path = os.path.join(
    MODELS_DIR, "biaffine-dependency-parser-ptb-2018.08.23.tar.gz")


def parse_args():
    parser = argparse.ArgumentParser() #创建解释器

    # Required parameters
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='/data1/SHENWZH/ABSA_online/data/semeval14',
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()




def xml2txt(file_path):
    '''
    Read the original xml file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    '''
    output = file_path.replace('.xml', '_text.txt')
    sent_list = []
    with open(file_path, 'rb') as f:
        raw = f.read()
        root = etree.fromstring(raw)
        for sentence in root:
            sent = sentence.find('text').text
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            if terms:
                sent_list.append(sent)
    with open(output, 'w') as f:
        for s in sent_list:
            f.write(s+'\n')
    print('processed', len(sent_list), 'of', file_path)


def text2docs(file_path, predictor):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    使用AllenNLP的预测器注释提取的txt文件中的句子。
    '''
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    docs = []
    print('Predicting dependency information...')
    for i in tqdm(range(len(sentences))):
        docs.append(predictor.predict(sentence=sentences[i]))

    return docs


def dependencies2format(doc):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    sentence = {}
    sentence['tokens'] = doc['words']
    sentence['tags'] = doc['pos']
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc['predicted_dependencies']
    predicted_heads = doc['predicted_heads']
    sentence['predicted_dependencies'] = doc['predicted_dependencies']
    sentence['predicted_heads'] = doc['predicted_heads']
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence


def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]
    return sentences


def syntaxInfo2json(sentences, origin_file):
    json_data = []
    tk = TreebankWordTokenizer()
    mismatch_counter = 0
    idx = 0
    with open(origin_file, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()#构造字典
            example["sentence"] = sentence.find('text').text
            
            # for RAN
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            
            example['tokens'] = sentences[idx]['tokens'] 
            example['tags'] = sentences[idx]['tags']
            example['predicted_dependencies'] = sentences[idx]['predicted_dependencies']
            example['predicted_heads'] = sentences[idx]['predicted_heads']
            example['dependencies'] = sentences[idx]['dependencies']
            # example['energy'] = sentences[idx]['energy']
            
            example["aspect_sentiment"] = []
            example['from_to'] = [] #left and right offset of the target word 

            for c in terms:
                if c.attrib['polarity'] == 'conflict':
                    continue
                target = c.attrib['term']
                example["aspect_sentiment"].append((target, c.attrib['polarity']))
                
                # index in strings, we want index in tokens
                left_index = int(c.attrib['from'])
                right_index = int(c.attrib['to'])

                left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))
                to_word_offset = len(tk.tokenize(example['sentence'][:right_index]))

                
                example['from_to'].append((left_word_offset,to_word_offset))
            if len(example['aspect_sentiment'])==0:
                idx += 1
                continue
            json_data.append(example)
            idx+=1
    extended_filename = origin_file.replace('.xml', '_biaffine_depparsed.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))
    print(idx)


def main():
    args = parse_args()

    predictor = Predictor.from_path(args.model_path)

    data = [('Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml'),
            ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml')]
    for train_file, test_file in data:
        # xml -> txt
        xml2txt(os.path.join(args.data_path, train_file))
        xml2txt(os.path.join(args.data_path, test_file))

        # txt -> json
        train_sentences = get_dependencies(
            os.path.join(args.data_path, train_file.replace('.xml', '_text.txt')), predictor)
        test_sentences = get_dependencies(os.path.join(
            args.data_path, test_file.replace('.xml', '_text.txt')), predictor)

        print(len(train_sentences), len(test_sentences))

        syntaxInfo2json(train_sentences, os.path.join(args.data_path, train_file))
        syntaxInfo2json(test_sentences, os.path.join(args.data_path, test_file))

if __name__ == "__main__":
    main()

from torch.nn.utils.rnn import pad_sequence
import argparse
import codecs
import json
import linecache
import logging
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from copy import copy, deepcopy

import nltk
import numpy as np
import simplejson as json
import torch
from allennlp.modules.elmo import batch_to_ids
from lxml import etree
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def load_datasets_and_vocabs(args):
    train, test = get_dataset(args.dataset_name)

    # Our model takes unrolled data, currently we don't consider the MAMS cases(future experiments)
    _, train_all_unrolled, _, _ = get_rolled_and_unrolled_data(train, args)
    _, test_all_unrolled, _, _ = get_rolled_and_unrolled_data(test, args)

    logger.info('****** After unrolling ******')
    logger.info('Train set size: %s', len(train_all_unrolled))
    logger.info('Test set size: %s,', len(test_all_unrolled))

    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(
        train_all_unrolled+test_all_unrolled, args)
    if args.embedding_type == 'glove':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.glove_embedding = embedding

    train_dataset = ASBA_Depparsed_Dataset(
        train_all_unrolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    test_dataset = ASBA_Depparsed_Dataset(
        test_all_unrolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)

    return train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab


def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


def get_dataset(dataset_name):
    '''
    Already preprocess the data and now they are in json format.(only for semeval14)
    Retrieve train and test set
    With a list of dict:
    e.g. {"sentence": "Boot time is super fast, around anywhere from 35 seconds to 1 minute.",
    "tokens": ["Boot", "time", "is", "super", "fast", ",", "around", "anywhere", "from", "35", "seconds", "to", "1", "minute", "."],
    "tags": ["NNP", "NN", "VBZ", "RB", "RB", ",", "RB", "RB", "IN", "CD", "NNS", "IN", "CD", "NN", "."],
    "predicted_dependencies": ["nn", "nsubj", "root", "advmod", "advmod", "punct", "advmod", "advmod", "prep", "num", "pobj", "prep", "num", "pobj", "punct"],
    "predicted_heads": [2, 3, 0, 5, 3, 5, 8, 5, 8, 11, 9, 9, 14, 12, 3],
    "dependencies": [["nn", 2, 1], ["nsubj", 3, 2], ["root", 0, 3], ["advmod", 5, 4], ["advmod", 3, 5], ["punct", 5, 6], ["advmod", 8, 7], ["advmod", 5, 8],
                    ["prep", 8, 9], ["num", 11, 10], ["pobj", 9, 11], ["prep", 9, 12], ["num", 14, 13], ["pobj", 12, 14], ["punct", 3, 15]],
    "aspect_sentiment": [["Boot time", "positive"]], "from_to": [[0, 2]]}
    '''
    rest_train = 'data/semeval14/Restaurants_Train_v2_biaffine_depparsed_with_energy.json'
    rest_test = 'data/semeval14/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json'

    laptop_train = 'data/semeval14/Laptop_Train_v2_biaffine_depparsed.json'
    laptop_test = 'data/semeval14/Laptops_Test_Gold_biaffine_depparsed.json'

    twitter_train = 'data/twitter/train_biaffine.json'
    twitter_test = 'data/twitter/test_biaffine.json'

    ds_train = {'rest': rest_train,
                'laptop': laptop_train, 'twitter': twitter_train}
    ds_test = {'rest': rest_test,
               'laptop': laptop_test, 'twitter': twitter_test}

    train = list(read_sentence_depparsed(ds_train[dataset_name]))
    logger.info('# Read %s Train set: %d', dataset_name, len(train))

    test = list(read_sentence_depparsed(ds_test[dataset_name]))
    logger.info("# Read %s Test set: %d", dataset_name, len(test))
    return train, test


def reshape_dependency_tree_new(as_start, as_end, dependencies, multi_hop=False, add_non_connect=False, tokens=None, max_hop = 5):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(2)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir


def reshape_dependency_tree(as_start, as_end, dependencies, add_2hop=False, add_non_connect=False, tokens=None):
    '''
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.

    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    # 2 hop
    if add_2hop:
        dep_idx_cp = dep_idx
        for i in dep_idx_cp:
            for dep in dependencies:
                # connect to i, not a punct
                if i == dep[1] - 1 and str(dep[0]) != 'punct':
                    # not root, not aspect
                    if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0:
                        if dep[2]-1 not in dep_idx:
                            dep_tag.append(dep[0])
                            dep_idx.append(dep[2] - 1)
                # connect to i, not a punct
                elif i == dep[2] - 1 and str(dep[0]) != 'punct':
                    # not root, not aspect
                    if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0:
                        if dep[1]-1 not in dep_idx:
                            dep_tag.append(dep[0])
                            dep_idx.append(dep[1] - 1)
    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir

def get_rolled_and_unrolled_data(input_data, args):
    '''
    In input_data, each sentence could have multiple aspects with different sentiments.
    Our method treats each sentence with one aspect at a time, so even for
    multi-aspect-multi-sentiment sentences, we will unroll them to single aspect sentence.

    Perform reshape_dependency_tree to each sentence with aspect

    return:
        all_rolled:
                a list of dict
                    {sentence, tokens, pos_tags, pos_class, aspects(list of aspects), sentiments(list of sentiments)
                    froms, tos, dep_tags, dep_index, dependencies}
        all_unrolled:
                unrolled, with aspect(single), sentiment(single) and so on...
        mixed_rolled:
                Multiple aspects and multiple sentiments, ROLLED.
        mixed_unrolled:
                Multiple aspects and multiple sentiments, UNROLLED.
    '''
    # A hand-picked set of part of speech tags that we see contributes to ABSA.
    opinionated_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
                        'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    all_rolled = []
    all_unrolled = []
    mixed_rolled = []
    mixed_unrolled = []

    unrolled = []
    mixed = []
    unrolled_ours = []
    mixed_ours = []

    # Make sure the tree is successfully built.
    zero_dep_counter = 0

    # Sentiment counters
    total_counter = defaultdict(int)
    mixed_counter = defaultdict(int)
    sentiments_lookup = {'negative': 0, 'positive': 1, 'neutral': 2}

    logger.info('*** Start processing data(unrolling and reshaping) ***')
    tree_samples = []
    # for seeking 'but' examples
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]
        aspects = []
        sentiments = []
        froms = []
        tos = []
        dep_tags = []
        dep_index = []
        dep_dirs = []

        # Classify based on POS-tags

        pos_class = e['tags']

        # Iterate through aspects in a sentence and reshape the dependency tree.
        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            # We would tokenize the aspect while at it.
            aspect = word_tokenize(aspect)
            sentiment = sentiments_lookup[e['aspect_sentiment'][i][1]]
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            aspects.append(aspect)
            sentiments.append(sentiment)
            froms.append(frm)
            tos.append(to)

            # Center on the aspect.
            dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(frm, to, e['dependencies'],
                                                       multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)

            # Because of tokenizer differences, aspect opsitions are off, so we find the index and try again.
            if len(dep_tag) == 0:
                zero_dep_counter += 1
                as_sent = e['aspect_sentiment'][i][0].split()
                as_start = e['tokens'].index(as_sent[0])
                # print(e['tokens'], e['aspect_sentiment'], e['dependencies'],as_sent[0])
                as_end = e['tokens'].index(
                    as_sent[-1]) if len(as_sent) > 1 else as_start + 1
                print("Debugging: as_start as_end ", as_start, as_end)
                dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(as_start, as_end, e['dependencies'],
                                                           multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)
                if len(dep_tag) == 0:  # for debugging
                    print("Debugging: zero_dep",
                          e['aspect_sentiment'][i][0], e['tokens'])
                    print("Debugging: ". e['dependencies'])
                else:
                    zero_dep_counter -= 1

            dep_tags.append(dep_tag)
            dep_index.append(dep_idx)
            dep_dirs.append(dep_dir)

            total_counter[e['aspect_sentiment'][i][1]] += 1

            # Unrolling
            all_unrolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'sentiment': sentiment,
                    'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir':dep_dir,'dependencies': e['dependencies']})


        # All sentences with multiple aspects and sentiments rolled.
        all_rolled.append(
            {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
             'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies']})

        # Ignore sentences with single aspect or no aspect
        if len(e['aspect_sentiment']) and len(set(map(lambda x: x[1], e['aspect_sentiment']))) > 1:
            mixed_rolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
                 'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies']})

            # Unrolling
            for i, as_sent in enumerate(e['aspect_sentiment']):
                mixed_counter[as_sent[1]] += 1
                mixed_unrolled.append(
                    {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspects[i], 'sentiment': sentiments[i],
                     'from': froms[i], 'to': tos[i], 'dep_tag': dep_tags[i], 'dep_idx': dep_index[i], 'dependencies': e['dependencies']})


    logger.info('Total sentiment counter: %s', total_counter)
    logger.info('Multi-Aspect-Multi-Sentiment counter: %s', mixed_counter)

    return all_rolled, all_unrolled, mixed_rolled, mixed_unrolled


def load_and_cache_vocabs(data, args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # Build or load word vocab and glove embeddings.
    # Elmo and bert have it's own vocab and embeddings.
    if args.embedding_type == 'glove':
        cached_word_vocab_file = os.path.join(
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',
                        args.dataset_name)
            word_vocab = build_text_vocab(data)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(
            args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.glove_dir)
            word_vecs = load_glove_embedding(
                word_vocab['itos'], args.glove_dir, 0.25, args.embedding_dim)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        word_vocab = None
        word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab


def load_glove_embedding(word_list, glove_dir, uniform_scale, dimension_size):
    glove_words = []
    with open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r',encoding='utf-8') as fopen:
        for line in fopen:
            glove_words.append(line.strip().split(' ')[0])
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(os.path.join(
                glove_dir, 'glove.840B.300d.txt'), word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(
                line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(
                np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors


def _default_unk_index():
    return 1


def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['sentence']
        counter.update(s)

    itos = ['[PAD]', '[UNK]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_pos_tag_vocab(data, vocab_size=1000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for d in data:
        tags = d['tags']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


# def build_dep_tag_vocab_energy():  # 47 in total, all tags plus pad and non-connect
#     '''
#     biaffine dep_tag Vocab : {0: 'punct', 1: 'prep', 2: 'pobj', 3: 'det', 4: 'nn',
#         5: 'nsubj', 6: 'amod', 7: 'root', 8: 'dobj', 9: 'aux', 10: 'advmod', 11: 'conj',
#         12: 'cc', 13: 'num', 14: 'poss', 15: 'ccomp', 16: 'dep', 17: 'xcomp', 18: 'mark',
#         19: 'cop', 20: 'number', 21: 'possessive', 22: 'rcmod', 23: 'auxpass', 24: 'appos',
#         25: 'nsubjpass', 26: 'advcl', 27: 'partmod', 28: 'pcomp', 29: 'neg', 30: 'tmod',
#         31: 'quantmod', 32: 'npadvmod', 33: 'prt', 34: 'infmod', 35: 'parataxis',
#         36: 'mwe', 37: 'expl', 38: 'acomp', 39: 'iobj', 40: 'csubj', 41: 'predet',
#         42: 'preconj', 43: 'discourse', 44: 'csubjpass'}
#     This is used in energy case.
#     '''
#     head_tags = {0: 'punct', 1: 'prep', 2: 'pobj', 3: 'det', 4: 'nn', 5: 'nsubj', 6: 'amod', 7: 'root', 8: 'dobj', 9: 'aux', 10: 'advmod', 11: 'conj', 12: 'cc', 13: 'num', 14: 'poss', 15: 'ccomp', 16: 'dep', 17: 'xcomp', 18: 'mark', 19: 'cop', 20: 'number', 21: 'possessive', 22: 'rcmod', 23: 'auxpass', 24: 'appos',
#                  25: 'nsubjpass', 26: 'advcl', 27: 'partmod', 28: 'pcomp', 29: 'neg', 30: 'tmod', 31: 'quantmod', 32: 'npadvmod', 33: 'prt', 34: 'infmod', 35: 'parataxis', 36: 'mwe', 37: 'expl', 38: 'acomp', 39: 'iobj', 40: 'csubj', 41: 'predet', 42: 'preconj', 43: 'discourse', 44: 'csubjpass', 45: '<pad>', 46: 'non-connect'}
#     itos = [head_tags[i] for i in range(len(head_tags))]
#     stoi = defaultdict()
#     stoi.update({tok: i for i, tok in enumerate(itos)})
#     return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


class ASBA_Depparsed_Dataset(Dataset):
    '''
    Convert examples to features, numericalize text to ids.
    data:
        -list of dict:
            keys: sentence, tags, pos_class, aspect, sentiment,
                predicted_dependencies, predicted_heads,
                from, to, dep_tag, dep_idx, dependencies, dep_dir

    After processing,
    data:
        sentence
        tags
        pos_class
        aspect
        sentiment
        from
        to
        dep_tag
        dep_idx
        dep_dir
        predicted_dependencies_ids
        predicted_heads
        dependencies
        sentence_ids
        aspect_ids
        tag_ids
        dep_tag_ids
        text_len
        aspect_len
        if bert:
            input_ids
            word_indexer

    Return from getitem:
        sentence_ids
        aspect_ids
        dep_tag_ids
        dep_dir_ids
        pos_class
        text_len
        aspect_len
        sentiment
        deprel
        dephead
        aspect_position
        if bert:
            input_ids
            word_indexer
            input_aspect_ids
            aspect_indexer
        or:
            input_cat_ids
            segment_ids
    '''

    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items = e['dep_tag_ids'], \
            e['pos_class'], e['text_len'], e['aspect_len'], e['sentiment'],\
            e['dep_rel_ids'], e['predicted_heads'], e['aspect_position'], e['dep_dir_ids']
        if self.args.embedding_type == 'glove':
            non_bert_items = e['sentence_ids'], e['aspect_ids']
            items_tensor = non_bert_items + items
            items_tensor = tuple(torch.tensor(t) for t in items_tensor)
        elif self.args.embedding_type == 'elmo':
            items_tensor = e['sentence_ids'], e['aspect_ids']
            items_tensor += tuple(torch.tensor(t) for t in items)
        else:  # bert
            if self.args.pure_bert:
                bert_items = e['input_cat_ids'], e['segment_ids']
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
            else:
                bert_items = e['input_ids'], e['word_indexer'], e['input_aspect_ids'], e['aspect_indexer'], e['input_cat_ids'], e['segment_ids']
                # segment_id
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
        return items_tensor

    def convert_features_bert(self, i):
        """
        BERT features.
        convert sentence to feature. 
        """
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        # tokenizer = self.args.tokenizer

        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []

        for word in self.data[i]['sentence']:
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            tokens.extend(word_tokens)
            # word_indexer is for indexing after bert, feature back to the length of original length.
            word_indexer.append(token_idx)

        # aspect
        for word in self.data[i]['aspect']:
            word_aspect_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(aspect_tokens)
            aspect_tokens.extend(word_aspect_tokens)
            aspect_indexer.append(token_idx)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens = [cls_token] + tokens + [sep_token]
        aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
        word_indexer = [i+1 for i in word_indexer]
        aspect_indexer = [i+1 for i in aspect_indexer]

        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
            aspect_tokens)

        # check len of word_indexer equals to len of sentence.
        assert len(word_indexer) == len(self.data[i]['sentence'])
        assert len(aspect_indexer) == len(self.data[i]['aspect'])

        # THE STEP:Zero-pad up to the sequence length, save to collate_fn.

        if self.args.pure_bert:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
        else:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
            self.data[i]['input_ids'] = input_ids
            self.data[i]['word_indexer'] = word_indexer
            self.data[i]['input_aspect_ids'] = input_aspect_ids
            self.data[i]['aspect_indexer'] = aspect_indexer

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):
            if self.args.embedding_type == 'glove':
                self.data[i]['sentence_ids'] = [self.word_vocab['stoi'][w]
                                                for w in self.data[i]['sentence']]
                self.data[i]['aspect_ids'] = [self.word_vocab['stoi'][w]
                                              for w in self.data[i]['aspect']]
            elif self.args.embedding_type == 'elmo':
                self.data[i]['sentence_ids'] = self.data[i]['sentence']
                self.data[i]['aspect_ids'] = self.data[i]['aspect']
            else:  # self.args.embedding_type == 'bert'
                self.convert_features_bert(i)

            self.data[i]['text_len'] = len(self.data[i]['sentence'])
            self.data[i]['aspect_position'] = [0] * self.data[i]['text_len']
            try:  # find the index of aspect in sentence
                for j in range(self.data[i]['from'], self.data[i]['to']):
                    self.data[i]['aspect_position'][j] = 1
            except:
                for term in self.data[i]['aspect']:
                    self.data[i]['aspect_position'][self.data[i]
                                                    ['sentence'].index(term)] = 1

            self.data[i]['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in self.data[i]['dep_tag']]
            self.data[i]['dep_dir_ids'] = [idx
                                           for idx in self.data[i]['dep_dir']]
            self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                             for w in self.data[i]['tags']]
            self.data[i]['aspect_len'] = len(self.data[i]['aspect'])

            self.data[i]['dep_rel_ids'] = [self.dep_tag_vocab['stoi'][r]
                                           for r in self.data[i]['predicted_dependencies']]


def my_collate(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.
    '''
    sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids = zip(
        *batch)  # from Dataset.__getitem__()
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    sentence_ids = pad_sequence(
        sentence_ids, batch_first=True, padding_value=0)
    aspect_ids = pad_sequence(aspect_ids, batch_first=True, padding_value=0)
    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    sentence_ids = sentence_ids[sorted_idx]
    aspect_ids = aspect_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids


def my_collate_elmo(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.
    The difference with my_collate is just padding method with sentence_ids and aspect_ids.
    '''
    sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids = zip(
        *batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    sentence_ids = batch_to_ids(sentence_ids)
    aspect_ids = batch_to_ids(aspect_ids)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    sentence_ids = sentence_ids[sorted_idx]
    aspect_ids = aspect_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids


# 11/7
def my_collate_pure_bert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    Pure Bert: cat text and aspect, cls to predict.
    Test indexing while at it?
    '''
    # sentence_ids, aspect_ids
    input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids = zip(
        *batch)  # from Dataset.__getitem__()

    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    input_cat_ids = pad_sequence(
        input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]

    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids


def my_collate_bert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    '''
    input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids,  dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids= zip(
        *batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_aspect_ids = pad_sequence(input_aspect_ids, batch_first=True, padding_value=0)
    input_cat_ids = pad_sequence(input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value =0)
    # indexer are padded with 1, for ...
    word_indexer = pad_sequence(word_indexer, batch_first=True, padding_value=1)
    aspect_indexer = pad_sequence(aspect_indexer, batch_first=True, padding_value=1)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    input_aspect_ids = input_aspect_ids[sorted_idx]
    word_indexer = word_indexer[sorted_idx]
    aspect_indexer = aspect_indexer[sorted_idx]
    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#from model_gcn import GAT, GCN, Rel_GAT
from model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits
from tree import *


class Aspect_Text_GAT_ours(nn.Module):
    """
    Full model in reshaped tree
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Text_GAT_ours, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim) #词典大小尺寸；嵌入维度default=300
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = [RelationAttention(in_dim = args.embedding_dim).to(args.device) for i in range(args.num_heads)]
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.hidden_size * 4

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]       #全连接层
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node  用邻近节点作为
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L',D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)      #按行求平均

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim = 1) # (N, H, D)
        dep_out = dep_out.mean(dim = 1) # (N, D)

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = torch.cat([dep_out,  gat_out], dim = 1) # (N, D')   #the second parament is gat_out originally
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

class Aspect_Text_GAT_only(nn.Module):
    """
    reshape tree in GAT only
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Text_GAT_only, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                                  bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)


        last_hidden_size = args.hidden_size * 2

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)


        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        ############################################################################################

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = gat_out # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


class Pure_Bert(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256):
        super(Pure_Bert, self).__init__()

        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        layers = [nn.Linear(
            config.hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, args.num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, token_type_ids):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids)
        # pool output is usually *not* a good summary of the semantic content of the input,
        # you're often better with averaging or poolin the sequence of hidden-states for the whole input sequence.
        pooled_output = outputs[1]
        # pooled_output = torch.mean(pooled_output, dim = 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class Aspect_Bert_GAT(nn.Module):
    '''
    R-GAT with bert
    '''

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Bert_GAT, self).__init__()
        self.args = args

        # Bert
        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)


        gcn_input_dim = args.embedding_dim

        # GAT
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]


        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        fmask[:,0] = 1
        outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
        feature_output = outputs[0] # (N, L, D)
        pool_out = outputs[1] #(N, D)

        # index select, back to original batched size.
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer)])

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D)


        feature_out = torch.cat([dep_out,  pool_out], dim=1)  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


class Without_relation_head(nn.Module):
    """
        model after changing
        only use gat_out for prediction
        """

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Without_relation_head, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)  # 词典大小尺寸；嵌入维度default=300
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim=gcn_input_dim, mem_dim=gcn_input_dim).to(args.device) for i in
                        range(args.num_heads)]  # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.hidden_size * 4

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]  # 全连接
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position,
                dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node  用邻近节点作为
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect)  # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature)  # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature)  # (N,L',D)

        aspect_feature = aspect_feature.mean(dim=1)  # (N, D)      #按行求平均

        ############################################################################################
        # do gat thing dep_tag为依存关系标签
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D)


        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature)  # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim=1))  # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = torch.cat([gat_out, gat_out], dim=1)  # (N, D')  the second parament is gat_out originally
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#图卷积层模块
def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class Rel_GAT(nn.Module):
    """ 
    Relation gat model, use the embedding of the edges to predict attention weight
    """

    def __init__(self, args, dep_rel_num,  hidden_size=64,  num_layers=2):
        super(Rel_GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        """
        # gat layer    图注意层
        # relation embedding, careful initialization?
        self.dep_rel_embed = nn.Embedding(
            dep_rel_num, args.dep_relation_embed_dim)
        nn.init.xavier_uniform_(self.dep_rel_embed.weight)

        # map rel_emb to logits. Naive attention on relations
        layers = [
            nn.Linear(args.dep_relation_embed_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.fcs = nn.Sequential(*layers)
  """
    def forward(self, adj, rel_adj, feature):
        denom = adj.sum(2).unsqueeze(2) + 1
        B, N = adj.size(0), adj.size(1)

        rel_adj_V = self.dep_rel_embed(
            rel_adj.view(B, -1))  # (batch_size, n*n, d)

        # gcn layer
        for l in range(self.num_layers):
            # relation based GAT, attention over relations
            
            if True:
                rel_adj_logits = self.fcs(rel_adj_V).squeeze(2)  # (batch_size, n*n)
            else:
                rel_adj_logits = self.A[l](rel_adj_V).squeeze(2)  # (batch_size, n*n)

            dmask = adj.view(B, -1)  # (batch_size, n*n)
            rel_adj_logits = F.softmax(
                mask_logits(rel_adj_logits, dmask), dim=1)
            rel_adj_logits = rel_adj_logits.view(
                *rel_adj.size())  # (batch_size, n, n)

            Ax = rel_adj_logits.bmm(feature)
            feature = self.dropout(Ax) if l < self.num_layers - 1 else Ax

        return feature


class GAT(nn.Module):
    """
    GAT module operated on graphs    GAT在图上的操作模块
    """

    def __init__(self, args, in_dim, hidden_size=64, mem_dim=300, num_layers=2):
        super(GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))

    def forward(self, adj,  feature):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            h = self.W[l](feature) # (B, N, D)
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N*N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            attention = F.softmax(mask_logits(e, dmask), dim=1)
            attention = attention.view(*adj.size())

            # original gat
            feature = attention.bmm(h)
            feature = self.dropout(feature) if l < self.num_layers - 1 else feature
            #####################################

        return feature


class GCN(nn.Module):
    """ 
    GCN module operated on graphs
    """

    def __init__(self, args, in_dim, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, feature):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.num_layers):
            Ax = adj.bmm(feature)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](feature)  # self loop
            AxW /= denom

            # gAxW = F.relu(AxW)
            gAxW = AxW
            feature = self.dropout(gAxW) if l < self.num_layers - 1 else gAxW
        return feature, mask

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class RelationAttention(nn.Module):
    def __init__(self, in_dim = 300, hidden_dim = 64):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        return out  # ([N, L])


class LinearAttention(nn.Module):
    '''
    re-implement of gat's attention
    '''
    def __init__(self, in_dim = 300, mem_dim = 300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(mem_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(1e-2)

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = self.linear(aspect_v) # (N, D)
        Q = Q.unsqueeze(1)  # (N, 1, D)
        Q = Q.expand_as(feature) # (N, L, D)
        Q = self.linear(Q) # (N, L, D)
        feature = self.linear(feature) # (N, L, D)

        att_feature = torch.cat([feature, Q], dim = 2) # (N, L, 2D)
        att_weight = self.fc(att_feature) # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, L, 1)
        att_weight = mask_logits(att_weight, dmask)  # (N, L ,1)

        attention = F.softmax(att_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)

        return out

#点积注意力表示attention（i，j）
class DotprodAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = aspect_v
        Q = Q.unsqueeze(2)  # (N, D, 1)
        dot_prod = torch.bmm(feature, Q)  # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, D, 1)

        attention_weight = mask_logits(dot_prod, dmask)  # (N, L ,1)
        #local operation

        attention = F.softmax(attention_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        # (N, D), ([N, L]), (N, L, 1)
        return out

class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class DepparseMultiHeadAttention(nn.Module):
    def __init__(self, h=6, Co=300, cat=True):
        super().__init__()
        self.hidden_size = Co // h
        self.h = h
        self.fc1 = nn.Linear(Co, Co)
        self.relu = nn.ReLU()
        self.fc2s = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(h)])
        self.cat = cat

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        nbatches = dep_tags_v.size(0)
        Q = self.fc1(dep_tags_v).view(nbatches, -1, self.h,
                                      self.hidden_size)  # [N, L, #heads, hidden_size]
        Q = self.relu(Q)
        Q = Q.transpose(0, 2)  # [#heads, L, N, hidden_size]
        Q = [l(q).squeeze(2).transpose(0, 1)
             for l, q in zip(self.fc2s, Q)]  # [N, L] * #heads
        # Q = Q.squeeze(2)
        Q = [F.softmax(mask_logits(q, dmask), dim=1).unsqueeze(2)
             for q in Q]  # [N, L, 1] * #heads

        # Q = Q.unsqueeze(2)
        if self.cat:
            out = torch.cat(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=1)
        else:
            out = torch.stack(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=2)
            out = torch.sum(out, dim=2)
        # out = out.squeeze(2)
        return out, Q[0]  # ([N, L]) one head

# coding=utf-8
import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

import numpy as np
import pandas as pd
import torch
from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs
from model import (Aspect_Text_GAT_ours,
                    Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only,Without_relation_head)
from trainer import train

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_name', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Choose absa dataset.')
    parser.add_argument('--output_dir', type=str, default='/data1/SHENWZH/ABSA_online/data/output-gcn',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')


    parser.add_argument('--cuda_id', type=str, default='3',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--glove_dir', type=str, default='data1/wordvec',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='data1/bert_base',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')

    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')


    parser.add_argument('--add_non_connect',  type= bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop',  type= bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type = int, default=4,
                        help='max number of hops')


    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')


    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_mem_dim', type=int, default=300,
                        help='Dimension of the W in GCN.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for GCN.')
    # GAT
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_new', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type = str, choices=['linear','dotprod','gcn'], default='dotprod',
                        help='The attention used for gat')

    parser.add_argument('--embedding_type', type=str,default='glove', choices=['glove','bert'])
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')

    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    
    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))
        


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer

    # Load datasets and vocabs
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab= load_datasets_and_vocabs(args)

    # Build Model
    # model = Aspect_Text_Multi_Syntax_Encoding(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
    if args.pure_bert:
        model = Pure_Bert(args)
    elif args.gat_bert:
        model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT + Bert
    elif args.gat_our:
        model = Aspect_Text_GAT_ours(args, dep_tag_vocab['len'], pos_tag_vocab['len']) # R-GAT with reshaped tree
    elif args.gat_new:
        model = Without_relation_head(args, dep_tag_vocab['len'], pos_tag_vocab['len']) # GAT with reshaped tree without realtion head
    else:
        model = Aspect_Text_GAT_only(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # original GAT with reshaped tree

    model.to(args.device)
    # Train
    _, _,  all_eval_results = train(args, train_dataset, model, test_dataset)

    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc']) 
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))


if __name__ == "__main__":
    main()
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import my_collate, my_collate_elmo, my_collate_pure_bert, my_collate_bert
from transformers import AdamW
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {  'sentence': batch[0],
                    'aspect': batch[1], # aspect token
                    'dep_tags': batch[2], # reshaped
                    'pos_class': batch[3],
                    'text_len': batch[4],
                    'aspect_len': batch[5],
                    'dep_rels': batch[7], # adj no-reshape
                    'dep_heads': batch[8],
                    'aspect_position': batch[9],
                    'dep_dirs': batch[10]
                    }
        labels = batch[6]
    else: # bert
        if args.pure_bert:
            # input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {  'input_ids': batch[0],
                        'token_type_ids': batch[1]}
            labels = batch[6]
        else:#bert&gat
            # input_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {  'input_ids': batch[0],
                        'input_aspect_ids': batch[2],
                        'word_indexer': batch[1],
                        'aspect_indexer': batch[3],
                        'input_cat_ids': batch[4],
                        'segment_ids': batch[5],
                        'dep_tags': batch[6],
                        'pos_class': batch[7],
                        'text_len': batch[8],
                        'aspect_len': batch[9],
                        'dep_rels': batch[11],
                        'dep_heads': batch[12],
                        'aspect_position': batch[13],
                        'dep_dirs': batch[14]}
            labels = batch[10]
    return inputs, labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        if args.pure_bert:
            return my_collate_pure_bert
        else:
            return my_collate_bert


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer


def train(args, train_dataset, model, test_dataset):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    
    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    #写入日志

    global_step = 0   #训练一个batch就＋1
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()  #将模型梯度参数置为零
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):   #遍历
            model.train()   #迭代
            batch = tuple(t.to(args.device) for t in batch)

            inputs, labels = get_input_from_batch(args, batch)  #输入
            logit = model(**inputs)  #模型的输出
            loss = F.cross_entropy(logit, labels)   #交叉熵损失

            if args.gradient_accumulation_steps > 1:   #梯度累计的情况
                loss = loss / args.gradient_accumulation_steps

            loss.backward()   #反向传播
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer.step()  #更新参数
                model.zero_grad()
                global_step += 1  #训练完一个batch

                # Log metrics  记录指标
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, test_dataset, model)
                    all_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss/global_step, all_eval_results


def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** Eval results *****')
        logger.info("  eval loss: %s", str(eval_loss))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    return results, eval_loss


def evaluate_badcase(args, eval_dataset, model, word_vocab):

    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)

    # Eval
    badcases = []
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)

        pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        label = int(labels.detach().cpu().numpy()[0])
        if pred != label:
            if args.embedding_type == 'bert':
                sent_ids = inputs['input_ids'][0].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = args.tokenizer.decode(sent_ids)
                case['aspect'] = args.tokenizer.decode(aspect_ids)
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)
            else:
                sent_ids = inputs['sentence'][0].detach().cpu().numpy()
                aspect_ids = inputs['aspect'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = ' '.join([word_vocab['itos'][i] for i in sent_ids])
                case['aspect'] = ' '.join([word_vocab['itos'][i] for i in aspect_ids])
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)

    return badcases


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)

"""
Basic operations on trees.
"""

from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

#迭代器
    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    head = head[:len_].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h-1]
                subj_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h-1]
                obj_ancestors.add(h-1)
                h = head[h-1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k: 0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h-1] is not None
                nodes[h-1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=False, self_loop=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret


def inputs_to_tree_reps(args, dep_heads, l, prune, subj_pos=None, obj_pos=None):
    '''
    Read the code.
    Inputs:
        head: predicted_heads
        l: (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1) get the text length in each sentence. For that we have text_len
            masks = torch.eq(words, 0)  (pads are zerors)
        maxlen: max(l), is just sentence.shape(1), we have that when batching.
        prune: we don't prune, set to -1.
        subj_pos: 
        obj_pos:

    head_to_tree: return the root.

    '''
    maxlen = max(l)
    dep_heads= dep_heads.cpu().numpy()
    if subj_pos:
        subj_pos = subj_pos.cpu().numpy() 
    if obj_pos:
        obj_pos = obj_pos.cpu().numpy()

    trees = [head_to_tree(dep_heads[i], l[i], prune,
                          None, None) for i in range(len(l))]
    adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(
        1, maxlen, maxlen) for tree in trees]
    adj = np.concatenate(adj, axis=0)
    adj = torch.from_numpy(adj)
    return adj


def inputs_to_deprel_adj(args, dep_heads, dep_rel, l, ):
    """
    Inputs:
        head: predicted_heads, used to form adj matrix. adj can be used to mask the logits of the rel_adj matrix.
        dep_rel: the corresponding relation. if a_ij = 1, r_ij = label. others will be masked, if set to ncon, then adj will not be used as mask.
        l: text_len

    """
    maxlen = max(l)
    dep_heads= dep_heads.cpu().numpy()
    dep_rel = dep_rel.cpu().numpy()
    
    trees = [head_rel_to_tree(dep_heads[i], dep_rel[i], l[i]) for i in range(len(l))]
    adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(
        1, maxlen, maxlen) for tree in trees]
    rel_adj = [tree_to_rel_adj(maxlen, tree, directed=False, self_loop=False).reshape(
        1, maxlen, maxlen) for tree in trees]
    
    adj = np.concatenate(adj, axis=0)
    adj = torch.from_numpy(adj)
    rel_adj = np.concatenate(rel_adj, axis=0)
    rel_adj = torch.from_numpy(rel_adj)
    return adj, rel_adj


def head_rel_to_tree(head, rel, len_):
    """
    Convert a sequence of head indexes into a tree object.将头部索引序列转化为树对象
    """
    head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].rel = rel[i]
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h-1].add_child(nodes[i])
    
    return root

def tree_to_rel_adj(sent_len, tree, directed=False, self_loop=True):
    """
    Convert a tree object to an (numpy) dep_rel labeled adjacency matrix.将树转化为dep_rel标记的邻接矩阵
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.int)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = t.rel
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret
