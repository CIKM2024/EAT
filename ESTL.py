# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers

class BaseModel(object):
    def __init__(self,
                 training_config,
                 name="CTR"):
        self.main_blocks = ['user_columns', 'item_columns']
        self.config = training_config

        self.gate_names = self.config.get("gates")
        self.gate_tensors = tf.stack([tf.constant(float(g)) for g in self.gate_names], axis=-1)
        self.default_gate_tensors = tf.stack(
            [tf.constant(0.0) for g in range(len(self.gate_names) - 1)] + [tf.constant(1.0)], axis=-1)

        t_trans_thre = self.config.get('trans_thres').split(',', -1)
        self.trans_thres = []
        for w in t_trans_thre:
            self.trans_thres.append(tf.constant(float(w)))


    def main_net(self, column_blocks, drop_out):
        # "main_embedding" represents the embedding vectors of inputs to the main net.
        self.main_net_ls = tf.concat(values=main_embedding, axis=1)
        # self.config.dnn_hidden_units: [1024, 512, 256]
        for layer_id, num_hidden_units in enumerate(self.config.dnn_hidden_units):
            self.main_net_ls = layers.fully_connected(
                self.main_net_ls,
                num_hidden_units,
                normalizer_fn=layers.batch_norm)
            self.main_net_ls = tf.layers.dropout(self.main_net_ls, rate=drop_out)
        return self.main_net_ls

    def gate_net(self, expert_num, gate_name, drop_out):
        # "gate_embedding" represents the embedding vectors of inputs to the gate net.
        self.gate_net_ls = tf.concat(values=gate_embedding, axis=1)
        for layer_id, num_hidden_units in enumerate(self.config.dnn_hidden_units):
            self.gate_net_ls = layers.fully_connected(
                self.gate_net_ls,
                num_hidden_units,
                normalizer_fn=layers.batch_norm)
            self.gate_net_ls = tf.layers.dropout(self.gate_net_ls, rate=drop_out)
            self.gate_net_ls = layers.fully_connected(
                self.gate_net_ls,
                expert_num,
                normalizer_fn=layers.batch_norm)
        self.gate_net_ls = tf.nn.softmax(self.gate_net_ls)
        return self.gate_net_ls

    def disc_layer(self):
        # "gate_embedding" represents the embedding vectors of inputs to the disc net.
        self.disc_net_ls = tf.concat(values=disc_embedding, axis=1)

        self.disc_net_ls = layers.fully_connected(
            self.disc_net_ls,
            2048,
            normalizer_fn=layers.batch_norm)

        for layer_id, num_hidden_units in enumerate(self.config.dnn_hidden_units):
            self.disc_net_ls = layers.fully_connected(
                self.disc_net_ls,
                num_hidden_units,
                normalizer_fn=layers.batch_norm)
            # self.gate_num = 2
            self.disc_net_ls = layers.fully_connected(
                self.disc_net_ls,
                self.gate_num,
                normalizer_fn=layers.batch_norm)
            self.disc_logits = tf.nn.softmax(self.disc_net_ls)
        return self.disc_net_ls


    def build_model(self):
        self.embedding_layer()
        self.disc_layer()
        self.build_estl_net()

    def build_estl_net(self):
        self.batch_size = tf.shape(self.label)[0]
        # Private-expert-net
        self.pr_experts = []
        for i, pr_name in enumerate(self.gate_names):
            expert_main = self.main_net(pr_name,self.drop_out[i])
            self.pr_experts.append(expert_main)
        # Public-expert-net
        pb_experts = []
        for i, pb_name in enumerate(self.expert_names):
            expert_main = self.main_net(pb_name,
                                        self.drop_out[i])
            pb_experts.append(expert_main)
        self.public_experts = tf.stack(pb_experts, axis=1)
        gates_w = []
        for i, gate_name in enumerate(self.gate_names):
            gates_w.append(self.gate_net(self.expert_num, 'gate_' + str(gate_name),self.drop_out[i]))
        self.gates_set = tf.stack(gates_w, axis=1)
        self.merge_logits(self.public_experts, self.gates_set, self.pr_experts)

    def merge_logits(self, public_experts, gates_set, private_experts):
        algates_expts_layer = tf.matmul(gates_set, public_experts)
        self.gates_out_layer = []
        for i, gate_name in enumerate(self.gate_names):
            private_public_merge = tf.concat([private_experts[i], algates_expts_layer[:,i,:]], axis=1)
            _net = layers.fully_connected(
                private_public_merge,
                64,
                normalizer_fn=layers.batch_norm)
            _net = tf.layers.dropout(_net, rate=self.drop_out[i], training=self.is_training,
                                     name="{}_hiddenlayer_dropout_0dp".format(self.name))
            _net = layers.fully_connected(
                _net,
                32,
                normalizer_fn=layers.batch_norm)
            _net = tf.layers.dropout(_net, rate=self.drop_out[i])
            main_logits = layers.linear(
                _net,
                1,
                biases_initializer=None)
            self.gates_out_layer.append(main_logits)

        self.stack_expert_logits = tf.stack(self.gates_out_layer, axis=1)
        self.logits = tf.reduce_sum(tf.multiply(self.stack_expert_logits, self.gate_mask), axis=1)









