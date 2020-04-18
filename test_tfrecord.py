# coding: utf-8

import sys

import tensorflow as tf

with open('/home/ndavid/vocab/wiki_bert_bpe_30000.txt') as inf:
    vocab = {i: token for i, token in enumerate(map(str.strip, inf))}

with open(f'{sys.argv[1]}.output', 'wt') as outf:
    for example_str in tf.python_io.tf_record_iterator(sys.argv[1], 'ZLIB'):
        example = tf.train.Example.FromString(example_str)
        input_ids = [x for x in example.features.feature['input_ids'].int64_list.value
                     if x != 0][1:-1]
        for input_id in input_ids:
            token = vocab[input_id]
            if token.startswith('##'):
                print(token[2:], file=outf, end='')
            else:
                print(f' {token}', file=outf, end='')
        print(file=outf)

# print(' '.join(map(str, input_ids)))
# print(' '.join(vocab[id] for id in input_ids))
# print(' '.join(f'{id}:{vocab[id]}' for id in input_ids))
