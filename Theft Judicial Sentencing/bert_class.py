# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import platform
import numpy as np
import pickle
import json

tf.logging.set_verbosity(tf.logging.WARN)

def getNumClasses(data_dir):
    tag_path = os.path.join(data_dir, "tags.txt")
    f = open(tag_path, 'r', encoding='utf-8')
    lines = f.readlines()
    label = []
    for line in lines:
        label.append(line.strip())
    f.close()
    return len(label)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.

        Note : text, label should be tokenization.convert_to_unicode()-e.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """
    
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
    
    
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, train_file):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, dev_file):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, test_file):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        infos = []
        with open(input_file, 'r', encoding = 'utf-8') as fin:
            for line in fin:
                infos.extend(json.loads(line))
        return infos
    
class MultilabelClassfier(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
    
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self, data_dir):
        """
            从data_dir中的 tags.txt 文件获取tags列表
        """
        tag_path = os.path.join(data_dir, "tags.txt")
        with open(tag_path, 'r', encoding='utf-8') as fin:
            labels = [label.strip() for label in fin]
        return labels

    def _create_examples(self, infos, set_type):
        examples = []
        for idx, data in enumerate(infos):
            guid = "%s-%s" % (set_type, idx)
            text_a = tokenization.convert_to_unicode(data['sentence'])
            labels = []
            for label in data['label']:
                label = tokenization.convert_to_unicode(label)
                labels.append(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = [0] * len(label_map)
    if len(example.label) > 0:
        for label in example.label:
            label_index = label_map[label]
            label_id[label_index] = 1

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %s)" % (str(example.label), str(label_id)))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        label_ids = feature.label_id
        features["label_ids"] = create_int_feature(label_ids)

        # features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, label_ids_len):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([label_ids_len], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model_original(bert_config, is_training, input_ids, input_mask, segment_ids,
                          labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)

class CNN_ATTENTION():
    def __init__(self, embedding, labels, num_classes, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0001):
        self.embedding_size = embedding_size
        self.embedded_chars = embedding
        self.num_classes = num_classes
        self.dropout_keep_prob = 0.5
        self.input_y = labels

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
                conv = tf.layers.conv1d(self.embedded_chars, num_filters, filter_size, name='conv1d')
                pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # self.probability = tf.nn.softmax(self.scores, name="probability")
            self.probabilities = tf.nn.sigmoid(self.scores, name="probabilities")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def pro(self):

        return self.loss, self.losses, self.scores, self.probabilities

    def attention(self, x_i, x, index):
        """
        Attention model for Neural Machine Translation
        :param x_i: the embedded input at time i
        :param x: the embedded input of all times(x_j of attentions)
        :param index: step of time
        """

        e_i = []
        c_i = []
        for output in x:
            output = tf.reshape(output, [-1, self.embedding_size])
            atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        # e_i = tf.exp(e_i)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.sequence_length, 1)

        # i!=j
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.embedding_size])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.sequence_length - 1, self.embedding_size])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i


def create_cnn_attention_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                               labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    embedding = model.get_sequence_output()
    # used = tf.sign(tf.abs(input_ids))
    # seq_length = tf.reduce_sum(used, reduction_indices=1)
    # seq_length = embedding.shape  #[batch_size, embedding_size]

    embedding_size = embedding.shape[-1].value

    labels = tf.cast(labels, tf.float32)

    cnn = CNN_ATTENTION(embedding=embedding, labels=labels, num_classes=num_labels,
                        embedding_size=embedding_size, filter_sizes=[2, 3, 4, 5], num_filters=128)

    loss, per_example_loss, logits, probabilities = cnn.pro()

    return (loss, per_example_loss, logits, probabilities)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.nn.sigmoid(logits)
        labels = tf.cast(labels, tf.float32)

        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.nn.sigmoid(logits)

                def multi_label_hot(predictions, threshold=0.5):
                    predictions = tf.cast(predictions, tf.float32)
                    threshold = float(threshold)
                    return tf.cast(tf.greater(predictions, threshold), tf.int64)

                one_hot_prediction = multi_label_hot(predictions)
                accuracy = tf.metrics.accuracy(tf.cast(one_hot_prediction, tf.int32), label_ids)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features

class BERT:
    def __init__(self,
                    bert_path = "./our_model/",
                    data_dir = "./data/all_data",
                    do_lower_case = True,
                    max_seq_length = 128,
                    output_dir = "./abl_model",
                    task_name = "multilabel",
                    use_tpu = False,
                    tpu_name = None,
                    tpu_zone = None,
                    gcp_project = None,
                    master = None,
                    save_checkpoints_steps = 100,
                    iterations_per_loop = 1000,
                    learning_rate = 2e-5,
                    num_train_epochs = 3,
                    train_batch_size = 32,
                    eval_batch_size = 8,
                    predict_batch_size = 8,
                    warmup_proportion = 0.1,
                    num_tpu_cores = 8
                    ):
        self.data_dir = data_dir
        self.do_lower_case = do_lower_case
        self.init_checkpoint = os.path.join(bert_path, "bert_model.ckpt")
        self.bert_config_file = os.path.join(bert_path, "bert_config.json")
        self.vocab_file = os.path.join(bert_path, "vocab.txt")
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.task_name = task_name
        self.use_tpu = use_tpu
        self.tpu_name = tpu_name
        self.tpu_zone = tpu_zone
        self.gcp_project = gcp_project
        self.master = master
        self.save_checkpoints_steps = save_checkpoints_steps
        self.iterations_per_loop = iterations_per_loop
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.warmup_proportion = warmup_proportion
        self.num_tpu_cores = num_tpu_cores
        
        self.train_examples = None
        self.num_train_steps = None
        self.num_warmup_steps = None
        self.bert_config = None
        self.label_list = None
        self.run_config = None
        self.read_config()

    def read_config(self):
        #tf.logging.set_verbosity(tf.logging.INFO)
    
        processors = {
            "multilabel": MultilabelClassfier,
        }
    
        tokenization.validate_case_matches_checkpoint(self.do_lower_case,
                                                      self.init_checkpoint)
    
        #if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        #    raise ValueError(
        #        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
    
        if self.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.max_seq_length, self.bert_config.max_position_embeddings))
    
        tf.gfile.MakeDirs(self.output_dir)
    
        task_name = self.task_name.lower()
    
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))
    
        self.processor = processors[task_name]()
    
        self.label_list = self.processor.get_labels(self.data_dir)
    
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
    
        tpu_cluster_resolver = None
        if self.use_tpu and self.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.tpu_name, zone=self.tpu_zone, project=self.gcp_project)
    
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.master,
            model_dir=self.output_dir,
            save_checkpoints_steps=self.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.iterations_per_loop,
                num_shards=self.num_tpu_cores,
                per_host_input_for_training=is_per_host))

    def get_estimator(self):
        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.label_list),
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=self.use_tpu,
            use_one_hot_embeddings=self.use_tpu)
    
        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.use_tpu,
            model_fn=model_fn,
            config=self.run_config,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            predict_batch_size=self.predict_batch_size)
        return estimator
    
    def train(self, train_filename, init_checkpoint = None):
        if init_checkpoint is not None:
            self.init_checkpoint = init_checkpoint
        print("INIT MODEL:", self.init_checkpoint)

        self.train_examples = self.processor.get_train_examples(self.data_dir, train_filename)
        self.num_train_steps = int(
            len(self.train_examples) / self.train_batch_size * self.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)
    
        estimator = self.get_estimator()

        train_file = os.path.join(self.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            self.train_examples, self.label_list, self.max_seq_length, self.tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(self.train_examples))
        tf.logging.info("  Batch size = %d", self.train_batch_size)
        tf.logging.info("  Num steps = %d", self.num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True,
            label_ids_len=getNumClasses(self.data_dir))
        estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

    def eval(self, dev_file, init_checkpoint):
        self.init_checkpoint = init_checkpoint
        estimator = self.get_estimator()
        eval_examples = self.processor.get_dev_examples(self.data_dir, dev_file)
        num_actual_eval_examples = len(eval_examples)
        if self.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % self.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())
    
        eval_file = os.path.join(self.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, self.label_list, self.max_seq_length, self.tokenizer, eval_file)
    
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", self.eval_batch_size)
    
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if self.use_tpu:
            assert len(eval_examples) % self.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // self.eval_batch_size)
    
        eval_drop_remainder = True if self.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            label_ids_len = getNumClasses(self.data_dir))
    
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    
        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        ret_info = []
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                print("Eval:  %s = %s" % (key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))
                ret_info.append((key, str(result[key])))
        return ret_info

    def predict(self, test_file, init_checkpoint):
        self.init_checkpoint = init_checkpoint
        estimator = self.get_estimator()
        predict_examples = self.processor.get_test_examples(self.data_dir, test_file)
        num_actual_predict_examples = len(predict_examples)
        if self.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % self.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())
    
        predict_file = os.path.join(self.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, self.label_list,
                                                self.max_seq_length, self.tokenizer,
                                                predict_file)
    
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", self.predict_batch_size)
    
        predict_drop_remainder = True if self.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            label_ids_len = getNumClasses(self.data_dir))
    
        result = estimator.predict(input_fn=predict_input_fn)
    
        output_predict_file = os.path.join(self.output_dir, "test_results.tsv")
        pred_labels = []
        pred_result = []
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            # output_file = './data/output.json'
            # output_file = open(output_file,'w')
            i = 0
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                # print(prediction)
                # print(type(prediction))
                prediction = prediction['probabilities']
                pred_result.append(prediction)
                predicted_labels = []
                predicted_scores = []
                count = 0
                for i, score in enumerate(prediction):
                    if score >= 0.5:
                        predicted_scores.append(score)
                        predicted_labels.append(i)
                        count += 1
                pred_labels.append(predicted_labels)
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)
        return pred_labels, pred_result

    def generate_pred_file(self, tags_list, inf_file, outf_path, pred_labels, result_pro):
        inf_path = os.path.join(self.data_dir, inf_file)
        with open(inf_path, 'r',encoding='utf-8') as inf, open(outf_path, 'w', encoding='utf-8') as outf:
            sentence_index = 0
            count = 0
            for line in inf.readlines():
                pre_doc = json.loads(line)
                predict_doc = []
                for ind in range(len(pre_doc)):
                    pred_sent = pre_doc[ind]
                    pred_sent['pro'] = []
                    pred_label = pred_labels[sentence_index]
    
                    label_names = []
                    for label in pred_label:
                        label_names.append(tags_list[label])
                    pred_sent['label'] = label_names
                    pred_sent['pro'] = result_pro[sentence_index].tolist()
                    predict_doc.append(pred_sent)
                    sentence_index += 1
                json.dump(predict_doc, outf, ensure_ascii=False)
                outf.write('\n')

def get_lastest_ckpt(path):
    with open(path) as fin:
        return fin.readline().strip().split(":")[1].strip().strip('"')

if __name__ == "__main__":

    tags_list = []
    with open('data/all_data/tags.txt', 'r', encoding='utf-8') as tagf:
      for line in tagf.readlines():
          tags_list.append(line.strip())

    model = BERT(bert_path = "./our_model", data_dir = "./data/all_data", output_dir = "./abl_model")

    model.train("50.json")
    lastest_ckpt = get_lastest_ckpt(model.output_dir + "/checkpoint")
    model.eval("40.json", model.output_dir + "/" + lastest_ckpt)
    pred_labels, result_pro = model.predict("10_new.json", model.output_dir + "/" + lastest_ckpt)#result_labels()

    inf_file = '10_new.json'
    outf_path = 'tmp/newoutput/output_22453_10_new.json'

    model.generate_pred_file(tags_list, inf_file, outf_path, pred_labels, result_pro)
