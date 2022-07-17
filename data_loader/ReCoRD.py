# -*- coding: utf-8 -*-
"""
Load ReCoRD dataset.
"""
import random
import torch
import spacy
import numpy as np
import ujson as json
from tqdm import tqdm
from codecs import open
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .config import *
from util.file_utils import pickle_dump_large_file, pickle_load_large_file


NLP = spacy.blank("en")


def word_tokenize(sent):
    doc = NLP(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def filter_func(config, example):
    return (len(example["context_tokens"]) > config.para_limit or
            len(example["ques_tokens"]) > config.ques_limit or
            (example["y2s"][0] - example["y1s"][0]) > config.ans_limit)  # !!!!!!! 多个ans？这里应该找到满足的那一个，而不是只看第一个


def get_examples(filename, data_type, word_counter, char_counter,
                 debug=False, debug_length=1):
    print("Generating {} examples...".format(data_type))
    examples = []
    meta = {}
    eval_examples = {}
    with open(filename, "r") as fh:
        source = json.load(fh)
        version = source["version"]
        meta["version"] = version
        meta["num_q"] = 0
        meta["num_q_noanswer"] = 0
        for article in tqdm(source["data"]):
            para = article['passage']
            qas = article['qas']
            context = para["text"].replace('\xa0', ' ')  # 跟 KT-NET 预处理代码学的
            entity_info = para['entities']
            context_tokens = word_tokenize(context)
            context_chars = [list(token) for token in context_tokens]
            spans = convert_idx(context, context_tokens)  # 左闭右开
            for token in context_tokens:
                word_counter[token] += len(qas)
                for char in token:
                    char_counter[char] += len(qas)
            for qa in qas:
                meta["num_q"] += 1
                ques = qa["query"].replace('\xa0', ' ')  # 跟 KT-NET 预处理代码学的
                ques_tokens = word_tokenize(ques)  # !!!!!!! @placeholder 这里需要特殊处理一下
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                y1s, y2s = [], []
                answer_texts = []
                answers = qa["answers"]
                if len(answers) == 0:
                    meta["num_q_noanswer"] += 1
                    continue
                for answer in answers:
                    answer_text = answer["text"]
                    answer_start = answer['start']
                    answer_end = answer_start + len(answer_text)
                    assert answer_end == answer['end'] + 1
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if answer_end > span[0] and answer_start < span[1]:  # 只要有交叉部分就算
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                context_tokens_ner = []
                for i, context_token in enumerate(context_tokens):
                    span = spans[i]
                    flag = False
                    for item in entity_info:
                        entity_start = item['start']
                        entity_end = item['end'] + 1
                        if entity_end > span[0] and entity_start < span[1]:  # 只要有交叉部分就算
                            if span[0] == entity_start and span[1] == entity_end:
                                context_tokens_ner.append('S')
                            elif span[0] == entity_start: 
                                context_tokens_ner.append('B')
                            elif span[1] == entity_end:
                                context_tokens_ner.append('E')
                            else:
                                context_tokens_ner.append('M')
                            flag = True
                            break
                    if not flag:
                        context_tokens_ner.append('O')
                example = {
                    "context_tokens": context_tokens,
                    "context_tokens_ner": context_tokens_ner, 
                    "context_chars": context_chars,
                    "ques_tokens": ques_tokens,
                    "ques_chars": ques_chars,
                    "y1s": y1s,
                    "y2s": y2s,
                    "id": meta["num_q"],
                    "context": context,
                    "spans": spans,
                    "answers": answer_texts,
                    "uuid": qa["id"]
                }
                examples.append(example)
                eval_examples[str(meta["num_q"])] = {
                    "context": context,
                    "spans": spans,
                    "answers": answer_texts,
                    "uuid": qa["id"]
                }
                if debug and meta["num_q"] >= debug_length:
                    return examples, meta
        # random.shuffle(examples)  # dataloader shuffle 了这里应该就不用 shuffle 了
        print("{} questions in total".format(len(examples)))
    return examples, meta, eval_examples


def get_embedding(counter, data_type,
                  emb_file=None, size=None, vec_size=None,
                  limit=-1, specials=["<PAD>", "<OOV>", "@placeholder", "<SOS>", "<EOS>"]):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                # if word in counter and counter[word] > limit: 
                #     embedding_dict[word] = vector
                # 同时考虑各种大小写情况，都加进去
                for each in [word, word.lower(), word.capitalize(), word.upper()]: 
                    if each in counter and counter[each] > limit and each not in embedding_dict:
                        embedding_dict[each] = vector
                        break
        print("{} / {} tokens have corresponding {} embedding vector".format(len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(len(filtered_elements)))

    token2idx_dict = {token: idx
                      for idx, token
                      in enumerate(embedding_dict.keys(), len(specials))}
    for i in range(len(specials)):
        token2idx_dict[specials[i]] = i
        embedding_dict[specials[i]] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def word2wid(word, word2idx_dict):
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2idx_dict:
            return word2idx_dict[each]
    return word2idx_dict["<OOV>"]


def char2cid(char, char2idx_dict):
    if char in char2idx_dict:
        return char2idx_dict[char]
    return char2idx_dict["<OOV>"]


def save(filepath, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)


def tokenize_context(context):
    context_split = context.split()
    context_tokenized = []
    tid2wid = {}
    wid = 0
    tid = 0
    for word in context_split:
        token = word_tokenize(word)
        context_tokenized.extend(token)
        for t in token:
            tid2wid[tid] = wid
            tid = tid + 1
        wid = wid + 1
    return context_tokenized, tid2wid


# def convert_to_features(config, context, question, word2idx_dict, char2idx_dict):
#     example = {}
#     context = context.replace("''", '" ').replace("``", '" ')
#     question = question.replace("''", '" ').replace("``", '" ')
#     example['context_tokens'], context_tid2wid = tokenize_context(context)
#     example['ques_tokens'] = word_tokenize(question)
#     example['context_chars'] = [
#         list(token) for token in example['context_tokens']]
#     example['ques_chars'] = [
#         list(token) for token in example['ques_tokens']]

#     para_limit = config.para_limit
#     ques_limit = config.ques_limit
#     char_limit = config.char_limit

#     def filter_func(example):
#         return (len(example["context_tokens"]) > para_limit or
#                 len(example["ques_tokens"]) > ques_limit)

#     if filter_func(example):
#         raise ValueError("Context/Questions lengths are over the limit")

#     context_idxs = np.zeros([para_limit], dtype=np.int32)
#     context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
#     ques_idxs = np.zeros([ques_limit], dtype=np.int32)
#     ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

#     for i, token in enumerate(example["context_tokens"]):
#         context_idxs[i] = word2wid(token, word2idx_dict)

#     for i, token in enumerate(example["ques_tokens"]):
#         ques_idxs[i] = word2wid(token, word2idx_dict)

#     for i, token in enumerate(example["context_chars"]):
#         for j, char in enumerate(token):
#             if j == char_limit:
#                 break
#             context_char_idxs[i, j] = char2cid(char, char2idx_dict)

#     for i, token in enumerate(example["ques_chars"]):
#         for j, char in enumerate(token):
#             if j == char_limit:
#                 break
#             ques_char_idxs[i, j] = char2cid(char, char2idx_dict)

#     return (context_idxs, context_char_idxs,
#             ques_idxs, ques_char_idxs, context_tid2wid)


def build_features(config, examples, meta, data_type, word2idx_dict, char2idx_dict, debug=False):
    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    examples_with_features = []
    for example in tqdm(examples):
        total_ += 1
        if filter_func(config, example):
            continue
        total += 1

        context_wids = np.ones([config.para_limit], dtype=np.int32) * word2idx_dict["<PAD>"]
        context_ner_ids = np.ones([config.para_limit], dtype=np.int32) * 0  # 先全部设置为 O
        context_cids = np.ones([config.para_limit, config.char_limit], dtype=np.int32) * char2idx_dict["<PAD>"]
        question_wids = np.ones([config.ques_limit], dtype=np.int32) * word2idx_dict["<PAD>"]
        question_cids = np.ones([config.ques_limit, config.char_limit], dtype=np.int32) * char2idx_dict["<PAD>"]
        y1 = np.zeros([config.para_limit], dtype=np.float32)
        y2 = np.zeros([config.para_limit], dtype=np.float32)

        for i, token in enumerate(example["context_tokens"]):
            context_wids[i] = word2wid(token, word2idx_dict)

        ner2id = {
            'O': 0, 
            'B': 1, 
            'M': 2, 
            'E': 3,
            'S': 4
        }
        for i, token in enumerate(example["context_tokens_ner"]):
            context_ner_ids[i] = ner2id[token]

        for i, token in enumerate(example["ques_tokens"]):
            question_wids[i] = word2wid(token, word2idx_dict)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == config.char_limit:
                    break
                context_cids[i, j] = char2cid(char, char2idx_dict)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == config.char_limit:
                    break
                question_cids[i, j] = char2cid(char, char2idx_dict)

        # !!!!!!! 用第一个 ans 作为 ans，应该改为用合适的那一个，或者用全部
        start, end = example["y1s"][0], example["y2s"][0]
        y1[start], y2[end] = 1.0, 1.0

        example["context_wids"] = context_wids
        example["context_cids"] = context_cids
        example["question_wids"] = question_wids
        example["question_cids"] = question_cids
        example["y1"] = start
        example["y2"] = end

        # 覆盖了原来的 example["context_tokens_ner"]，现在是 id
        example["context_tokens_ner"] = context_ner_ids

        # don't store unnecessary properties to save shared memory when loading data
        example["spans"] = None
        example["context_chars"] = None
        example["ques_chars"] = None

        # example["context_tokens"] = None  # 此处保留 token，之后会用到
        if len(example["context_tokens"]) < config.para_limit: 
            example["context_tokens"].extend(["<PAD>" for i in range(config.para_limit - len(example["context_tokens"]))])

        # example["ques_tokens"] = None  # 此处保留 token，之后会用到
        if len(example["ques_tokens"]) < config.ques_limit: 
            example["ques_tokens"].extend(["<PAD>" for i in range(config.ques_limit - len(example["ques_tokens"]))])

        examples_with_features.append(example)

    print("Built {} / {} instances of features in total".format(total, total_))
    meta["num_q_filtered"] = total
    return examples_with_features, meta


def prepro(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_meta, train_eval = get_examples(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_meta, dev_eval = get_examples(
        config.dev_file, "dev", word_counter, char_counter)

    word_emb_file = config.glove_word_file
    word_emb_size = config.glove_word_size
    word_emb_dim = config.glove_dim
    pretrained_char = config.pretrained_char
    char_emb_file = config.glove_char_file if pretrained_char else None
    char_emb_size = config.glove_char_size if pretrained_char else None
    char_emb_dim = config.glove_dim if pretrained_char else config.char_dim
    # word_emb_mat, word2idx_dict = get_embedding(
    #     word_counter, "word", emb_file=word_emb_file,
    #     size=word_emb_size, vec_size=word_emb_dim)
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file,
        size=word_emb_size, vec_size=word_emb_dim, limit=0)  # 试试更高的词频 limit，试过了，5 和 3 都没有 0 效果好
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file,
        size=char_emb_size, vec_size=char_emb_dim)

    train_examples, train_meta = build_features(
        config, train_examples, train_meta, "train",
        word2idx_dict, char2idx_dict)
    dev_examples, dev_meta = build_features(
        config, dev_examples, dev_meta, "dev",
        word2idx_dict, char2idx_dict)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
    save(config.train_examples_file, train_examples, message="train examples")
    save(config.dev_examples_file, dev_examples, message="dev examples")
    save(config.train_meta_file, train_meta, message="train meta")
    save(config.dev_meta_file, dev_meta, message="dev meta")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")


class ReCoRD(Dataset):

    def __init__(self, examples_file, data_ratio=1.0):
        self.examples = pickle_load_large_file(examples_file)
        used_num = int(len(self.examples) * data_ratio)
        self.examples = self.examples[:used_num]
        self.num = len(self.examples)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return (self.examples[idx]["context_wids"],
                self.examples[idx]["context_cids"],
                self.examples[idx]["question_wids"],
                self.examples[idx]["question_cids"],
                self.examples[idx]["y1"],
                self.examples[idx]["y2"],
                self.examples[idx]["y1s"],
                self.examples[idx]["y2s"],
                self.examples[idx]["id"],
                self.examples[idx]["context_tokens"],
                self.examples[idx]["ques_tokens"], 
                self.examples[idx]["context_tokens_ner"]
        )


def collate(data):
    Cwid, Ccid, Qwid, Qcid, y1, y2, y1s, y2s, id, context_tokens, ques_tokens, Cnid = zip(*data)
    Cwid = torch.tensor(Cwid).long()
    Ccid = torch.tensor(Ccid).long()
    Qwid = torch.tensor(Qwid).long()
    Qcid = torch.tensor(Qcid).long()
    y1 = torch.from_numpy(np.array(y1)).long()
    y2 = torch.from_numpy(np.array(y2)).long()
    id = torch.from_numpy(np.array(id)).long()
    Cnid = torch.tensor(Cnid).long()
    return Cwid, Ccid, Qwid, Qcid, y1, y2, y1s, y2s, id, context_tokens, ques_tokens, Cnid


def get_loader(examples_file, batch_size, shuffle=True, data_ratio=1.0):
    dataset = ReCoRD(examples_file, data_ratio)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # num_works > 0 may cause RequestRefused error
        collate_fn=collate)
    return data_loader
