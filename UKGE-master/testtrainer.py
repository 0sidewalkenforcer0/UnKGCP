''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from os.path import join

import param

import sys

if '../src' not in sys.path:
    sys.path.append('../src')

import numpy as np
import tensorflow as tf
import time
from data import BatchLoader

from utils import vec_length
from list import ModelList
from models import UKGE_logi_TF, UKGE_rect_TF
from testtesters import UKGE_logi_Tester, UKGE_rect_Tester




class Trainer(object):
    def __init__(self):
        self.batch_size = 128
        self.dim = 64
        self.this_data = None
        self.tf_parts = None
        self.save_path = 'this-distmult.ckpt'
        self.data_save_path = 'this-data.bin'
        self.file_val = ""
        self.L1 = False

    def build(self, data_obj, save_dir,
              model_save='model.bin',
              data_save='data.bin'):
        """
        All files are stored in save_dir.
        output files:
        1. tf model
        2. this_data (Data())
        3. training_loss.csv, val_loss.csv
        :param model_save: filename for model
        :param data_save: filename for self.this_data
        :param knn_neg: use kNN negative sampling
        :return:
        """
        self.verbose = param.verbose  # print extra information
        self.this_data = data_obj
        self.dim = self.this_data.dim = param.dim
        self.batch_size = self.this_data.batch_size = param.batch_size
        self.neg_per_positive = param.neg_per_pos
        self.reg_scale = param.reg_scale

        self.batchloader = BatchLoader(self.this_data, self.batch_size, self.neg_per_positive)

        self.p_neg = param.p_neg
        self.p_psl = param.p_psl

        # paths for saving
        self.save_dir = save_dir
        self.save_path = join(save_dir, model_save)  # tf model
        self.data_save_path = join(save_dir, data_save)  # this_data (Data())
        self.train_loss_path = join(save_dir, 'trainig_loss.csv')
        self.val_loss_path = join(save_dir, 'val_loss.csv')

        print('Now using model: ', param.whichmodel)

        self.whichmodel = param.whichmodel

        self.build_tf_parts()  # could be overrided

    def build_tf_parts(self):
        """
        Build tfparts (model) and validator.
        Different for every model.
        :return:
        """
        if self.whichmodel == ModelList.LOGI:
            self.tf_parts = UKGE_logi_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg)
            self.validator = UKGE_logi_Tester()

        elif self.whichmodel == ModelList.RECT:
            self.tf_parts = UKGE_rect_TF(num_rels=self.this_data.num_rels(),
                                         num_cons=self.this_data.num_cons(),
                                         dim=self.dim,
                                         batch_size=self.batch_size,
                                         neg_per_positive=self.neg_per_positive, p_neg=self.p_neg, reg_scale=self.reg_scale)
            self.validator = UKGE_rect_Tester()



    def gen_batch(self, forever=False, shuffle=True, negsampler=None):
        """
        :param ht_embedding: for kNN negative sampling
        :return:
        """
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples  # np.float64 [[h,r,t,w]]
            if shuffle:
                np.random.seed(param.seed)
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):

                batch = triples[i: i + self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_size

                h_batch, r_batch, t_batch, w_batch = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]

                hrt_batch = batch[:, 0:3].astype(int)

                all_neg_hn_batch = self.this_data.corrupt_batch(hrt_batch, self.neg_per_positive, "h")
                all_neg_tn_batch = self.this_data.corrupt_batch(hrt_batch, self.neg_per_positive, "t")

                neg_hn_batch, neg_rel_hn_batch, \
                neg_t_batch, neg_h_batch, \
                neg_rel_tn_batch, neg_tn_batch \
                    = all_neg_hn_batch[:, :, 0], \
                      all_neg_hn_batch[:, :, 1], \
                      all_neg_hn_batch[:, :, 2], \
                      all_neg_tn_batch[:, :, 0], \
                      all_neg_tn_batch[:, :, 1], \
                      all_neg_tn_batch[:, :, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), w_batch.astype(
                    np.float32), \
                      neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), \
                      neg_t_batch.astype(np.int64), neg_h_batch.astype(np.int64), \
                      neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)
            if not forever:
                break

    def train(self, epochs=20, save_every_epoch=10, lr=0.001, data_dir="", wandb=None):
        start_time = time.time()

        best_metric = {
            'valid_mse': 1,
            'valid_mse_neg': 1,
            'valid_mse_combined': 1,
            'test_mse': 1,
            'test_mae': 100,
            'test_mse_with_neg': 1,
            'test_mae_with_neg': 100
        }
        last_best_metric = best_metric.copy()
        last_best_epoch = 0

        sess = tf.Session()  # show device info
        sess.run(tf.global_variables_initializer())

        num_batch = self.this_data.triples.shape[0] // self.batch_size
        print('Number of batches per epoch: %d' % num_batch)


        train_losses = []  # [[every epoch, loss]]
        val_losses = []  # [[saver epoch, loss]]

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train1epoch(sess, num_batch, lr, epoch)
            train_losses.append([epoch, epoch_loss])

            if np.isnan(epoch_loss):
                print("Nan loss. Training collapsed.")
                return

            if epoch % save_every_epoch == 0:
                # validation error
                val_loss, val_loss_neg = self.get_val_loss(epoch, sess)  # loss for testing triples and negative samples
                val_loss_combined = (val_loss + val_loss_neg) / 2
                wandb.log({'Valid MSE': val_loss, 'Valid MSE neg': val_loss_neg, 'Valid MSE combined': val_loss_combined})

                # save and print metrics
                val_losses.append(['valid', val_loss, val_loss_neg, val_loss_combined])
                self.save_loss(train_losses, self.train_loss_path, columns=['epoch', 'training_loss'])
                self.save_loss(val_losses, self.val_loss_path,
                               columns=['val_epoch', 'mse', 'mse_neg', 'mse_combined'])

                if val_loss < best_metric['valid_mse']:
                    best_metric['valid_mse'] = val_loss

                if val_loss_neg < best_metric['valid_mse_neg']:
                    best_metric['valid_mse_neg'] = val_loss_neg

                if val_loss_combined < best_metric['valid_mse_combined']:
                    best_metric['valid_mse_combined'] = val_loss_combined

                # early stopping
                if epoch >= 1 and best_metric['valid_mse_combined'] >= last_best_metric['valid_mse_combined']:  # no improvement or already overfit
                    print('epoch', epoch, 'last_best_epoch', last_best_epoch)
                    if epoch - last_best_epoch >= 200:  # patience
                        print('***best epoch:***', last_best_epoch)
                        print('***best metric:***', last_best_metric)
                        wandb.log({'epoch': last_best_epoch})

                        # run.finish()  # end wandb watch
                        break
                else:
                    test_mse, test_mae = self.get_test_loss(epoch, sess)
                    wandb.log({'Test MSE': test_mse, 'Test MAE': test_mae})

                    test_mse_with_neg, test_mae_with_neg = self.get_test_with_neg_loss(epoch, sess)
                    wandb.log({'Test MSE with neg': test_mse_with_neg, 'Test MAE with neg': test_mae_with_neg})

                    if test_mse < best_metric['test_mse']:
                        best_metric['test_mse'] = test_mse
                    
                    if test_mae < best_metric['test_mae']:
                        best_metric['test_mae'] = test_mae

                    if test_mse_with_neg < best_metric['test_mse_with_neg']:
                        best_metric['test_mse_with_neg'] = test_mse_with_neg

                    if test_mae_with_neg < best_metric['test_mae_with_neg']:
                        best_metric['test_mae_with_neg'] = test_mae_with_neg

                    last_best_metric = best_metric.copy()
                    last_best_epoch = epoch
                    this_save_path = self.tf_parts._saver.save(sess, self.save_path, global_step=epoch)
                    self.this_data.save(self.data_save_path)
                    with sess.as_default():
                        ht_embeddings = self.tf_parts._ht.eval()
                        r_embeddings = self.tf_parts._r.eval()
                    print('best_epoch', last_best_epoch)
            
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))

        #test_mse, test_mae = self.get_test_loss(epoch, sess)
        #test_mse_with_neg, test_mae_with_neg = self.get_test_with_neg_loss(epoch, sess)
        #val_losses.append(['test', test_mse, test_mse_with_neg, test_mae, test_mae_with_neg])#, mean_ndcg, mean_exp_ndcg
        #self.save_loss(val_losses, self.val_loss_path, columns=['val_epoch', 'mse', 'mse_with_neg', 'mae', 'mae_with_neg'])#, 'ndcg(linear)', 'ndcg(exp)'])
        
        print("Model saved in file: %s" % this_save_path)
        sess.close()
        return ht_embeddings, r_embeddings

    def get_val_loss(self, epoch, sess):
        # validation error

        self.validator.build_by_var(self.this_data.val_triples, self.tf_parts, self.this_data, sess=sess)

        '''if not hasattr(self.validator, 'hr_map'):
            self.validator.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv',
                                                                          'test.tsv'])
        if not hasattr(self.validator, 'hr_map_sub'):
            hr_map200 = self.validator.get_fixed_hr(n=200)  # use smaller size for faster validation
        else:
            hr_map200 = self.validator.hr_map_sub'''

        #mean_ndcg, mean_exp_ndcg = self.validator.mean_ndcg(hr_map200)

        # metrics: mse
        mse, _ = self.validator.get_mse(save_dir=self.save_dir, epoch=epoch, is_test=False)
        neg_per_positive = int(10)
        mse_neg = self.validator.get_mse_neg(neg_per_positive)
        
        
        errors = param.errors
        errors_neg = param.errors_neg
        errors_combined = np.abs(np.concatenate([errors, errors_neg]))
        n = len(errors_combined)
        conf_level = np.ceil((n+1)*(1-0.1))/n
        qhat_combined = np.quantile(errors_combined, conf_level, interpolation="higher")

        '''plt.hist(errors_combined, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel("Error Values")
        plt.ylabel("Frequency")
        plt.title("Histogram of Errors Combined")
        plt.show()'''
        
        param.qhat_combined = qhat_combined
        #print('qhat_combined: %f' % qhat_combined)
        
        ## Calculate normalized nonconf scores
        normalized_nonconf_scores_combined = np.concatenate((param.normalizing_nonconf_scores, param.normalizing_errors_neg))
        
        normalizing_conf_level = np.ceil((len(normalized_nonconf_scores_combined)+1)*(1-0.1))/len(normalized_nonconf_scores_combined)
        param.normalizing_qhat_combined = np.quantile(normalized_nonconf_scores_combined, normalizing_conf_level, interpolation="higher")
        
        ## Calculate normalized nonconf scores after mapping for CN15K
        normalized_nonconf_scores_combined_mapping = np.concatenate((param.normalizing_nonconf_scores_mapping, param.normalizing_errors_neg_mapping))
        
        normalizing_conf_level_mapping = np.ceil((len(normalized_nonconf_scores_combined_mapping)+1)*(1-0.1))/len(normalized_nonconf_scores_combined_mapping)
        param.normalizing_qhat_combined_mapping = np.quantile(normalized_nonconf_scores_combined_mapping, normalizing_conf_level_mapping, interpolation="higher")
        return mse, mse_neg#, mean_ndcg, mean_exp_ndcg

    def get_test_loss(self, epoch, sess):
        # test error

        self.validator.build_by_var(self.this_data.test_triples, self.tf_parts, self.this_data, sess=sess)

        '''if not hasattr(self.validator, 'hr_map'):
            self.validator.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv',
                                                                          'test.tsv'])
        if not hasattr(self.validator, 'hr_map_sub'):
            hr_map200 = self.validator.get_fixed_hr(n=200)  # use smaller size for faster validation
        else:
            hr_map200 = self.validator.hr_map_sub'''

        #mean_ndcg, mean_exp_ndcg = self.validator.mean_ndcg(hr_map200)

        # metrics: mse
        mse, mae = self.validator.get_mse(save_dir=self.save_dir, epoch=epoch, is_test=True)
        #mse_neg = self.validator.get_mse_neg(self.neg_per_positive)
        return mse, mae #, mean_ndcg, mean_exp_ndcg

    def get_test_with_neg_loss(self, epoch, sess):
        # test with neg error

        self.validator.build_by_var(self.this_data.test_triples_with_neg, self.tf_parts, self.this_data, sess=sess)

        '''if not hasattr(self.validator, 'hr_map'):
            self.validator.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv',
                                                                          'test.tsv'])
        if not hasattr(self.validator, 'hr_map_sub'):
            hr_map200 = self.validator.get_fixed_hr(n=200)  # use smaller size for faster validation
        else:
            hr_map200 = self.validator.hr_map_sub'''

        #mean_ndcg, mean_exp_ndcg = self.validator.mean_ndcg(hr_map200)

        # metrics: mse
        mse_with_neg, mae_with_neg = self.validator.get_mse(save_dir=self.save_dir, epoch=epoch, is_test=True)
        #mse_neg = self.validator.get_mse_neg(self.neg_per_positive)
        return mse_with_neg, mae_with_neg #, mean_ndcg, mean_exp_ndcg

    def save_loss(self, losses, filename, columns):
        df = pd.DataFrame(losses, columns=columns)
        if 'val_epoch' in df.columns and (df['val_epoch'] == 'test').any():
            print(df[df['val_epoch'] == 'test'])
        else:
            print(df.tail(5))
        df.to_csv(filename, index=False)

    def train1epoch(self, sess, num_batch, lr, epoch):
        batch_time = 0

        epoch_batches = self.batchloader.gen_batch(forever=True)

        epoch_loss = []

        for batch_id in range(num_batch):

            batch = next(epoch_batches)
            A_h_index, A_r_index, A_t_index, A_w, \
            A_neg_hn_index, A_neg_rel_hn_index, \
            A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch

            time00 = time.time()
            soft_h_index, soft_r_index, soft_t_index, soft_w_index = self.batchloader.gen_psl_samples()  # length: param.n_psl
            batch_time += time.time() - time00

            _, gradient, batch_loss, psl_mse, mse_pos, mse_neg, main_loss, psl_prob, psl_mse_each, rule_prior, psl_loss, f_score_tn, f_score_hn= sess.run(
                [self.tf_parts._train_op, self.tf_parts._gradient,
                 self.tf_parts._A_loss, self.tf_parts.psl_mse, self.tf_parts._f_score_h, self.tf_parts._f_score_hn,
                 self.tf_parts.main_loss, self.tf_parts.psl_prob, self.tf_parts.psl_error_each,
                 self.tf_parts.prior_psl0, self.tf_parts.psl_loss, self.tf_parts._f_score_tn, self.tf_parts._f_score_hn],
                feed_dict={self.tf_parts._A_h_index: A_h_index,
                           self.tf_parts._A_r_index: A_r_index,
                           self.tf_parts._A_t_index: A_t_index,
                           self.tf_parts._A_w: A_w,
                           self.tf_parts._A_neg_hn_index: A_neg_hn_index,
                           self.tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                           self.tf_parts._A_neg_t_index: A_neg_t_index,
                           self.tf_parts._A_neg_h_index: A_neg_h_index,
                           self.tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                           self.tf_parts._A_neg_tn_index: A_neg_tn_index,
                           self.tf_parts._soft_h_index: soft_h_index,
                           self.tf_parts._soft_r_index: soft_r_index,
                           self.tf_parts._soft_t_index: soft_t_index,
                           self.tf_parts._soft_w: soft_w_index,
                           self.tf_parts._lr: lr
                           })
            #print('psl_mse: %f' % psl_mse)
            #print('psl_prob: %f' % psl_prob)
            #print('psl_mse_each: %s' % psl_mse_each)
            #print('rule_prior: %s' % rule_prior)
            '''print('psl_loss: %f' % psl_loss)
            print('main_loss: %f' % main_loss)
            print('batch_loss: %f' % batch_loss)
            print('f_score_tn: %s' % f_score_tn)
            print('f_score_hn: %s' % f_score_hn)
            print('this_loss: %f' % this_loss)
            print('regularizer: %f' % regularizer)'''

            param.prior_psl = rule_prior
            epoch_loss.append(batch_loss)

            if ((batch_id + 1) % 50 == 0) or batch_id == num_batch - 1:
                print('process: %d / %d. Epoch %d' % (batch_id + 1, num_batch, epoch))

        this_total_loss = np.sum(epoch_loss) / len(epoch_loss)
        print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
        # print('MSE on positive instances: %f, MSE on negative samples: %f' % (np.mean(mse_pos), np.mean(mse_neg)))

        return this_total_loss