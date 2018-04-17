import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.append('utils')
from nets import *
from datas import *
class auto_encoder():
    def __init__(self, model,data):
                self.model = model
                # self.model_q = model_q
                self.data=data
                self.size = self.data.size
                self.channel = self.data.channel
                self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
                self.label = tf.placeholder(tf.float32,shape=[None,10])
                self.q,self.g=model(self.X)
                # self.g=model_g(self.q)
                self.q=self.q[:,0:10]
                print(self.q.shape)                                                                          
                self.q_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.q,labels=self.label))
                self.g_loss = tf.reduce_mean(tf.square(self.X-self.g)/tf.abs(self.g))
                self.q_solver= tf.train.AdamOptimizer().minimize(self.q_loss, var_list=self.model.vars_q)
                self.g_solver= tf.train.AdamOptimizer().minimize(self.g_loss, var_list=self.model.vars_g+self.model.vars_q)
                tf.summary.scalar('q_loss',self.q_loss)
                tf.summary.scalar('g_loss',self.g_loss)
                for var in self.model.vars_q:
                        tf.summary.histogram(var.name,var)
                for var in self.model.vars_g:
                        tf.summary.histogram(var.name,var)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
                # gpu_options = tf.GPUOptions()
                self.saver = tf.train.Saver()
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 512):
        fig_count = 0
        train_writer = tf.summary.FileWriter(sample_dir + '/train',self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        for epoch in range(training_epoches):
            X_b, label= self.data(batch_size)
            k=1
            for _ in range(k):
                summary,_=self.sess.run(
					[merged,self.g_solver],
					feed_dict={self.X: X_b, self.label: label})
            # _=self.sess.run(self.q_solver,feed_dict={self.X: X_b, self.label: label})
            train_writer.add_summary(summary,epoch)
			# train_writer.add_summary(summary[1],epoch)
			# save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                q_loss_curr, g_loss_curr = self.sess.run(
						[self.q_loss, self.g_loss],
					    feed_dict={self.X: X_b, self.label: label})
                print('Iter: {}; G_loss: {:.4}; Q_loss: {:.4}'.format(epoch, g_loss_curr, q_loss_curr))
                
                if epoch % 1000 == 0:
                    samples = self.sess.run(self.g, feed_dict={self.X: X_b, self.label: label})
                    fig = self.data.data2fig(samples[:16])
                    plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)
                if epoch % 2000 == 0:
                    self.saver.save(self.sess, os.path.join(ckpt_dir, "autoencoder.ckpt"))

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	# save generated images
    sample_dir = 'Samples/autoencoder'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    autoencoder1=Auto_encoder()
    autoencoder2=Auto_encoder_g()

    data = mnist()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
	# run
    infogan = auto_encoder(autoencoder1,data)
    infogan.train(sample_dir)