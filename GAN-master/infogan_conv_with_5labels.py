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

def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

# def sample_c(m, n, ind=-1):
# 	c = np.zeros([m,n])
# 	for i in range(m):
# 		if ind<0:
# 			ind = np.random.randint(10)
# 		c[i,i%10] = 1
# 	return c
def sample_c(m, n, ind=-1):
	# c = np.zeros([m,n])
	c= np.zeros([m,n])
	for i in range(m):
		if ind<0:
			ind = np.random.randint(10)
		# c[i,i%10] = 1.
		for j in range(0,n):
			c[i,j]=np.random.uniform(-2.,2.)
	return c

def concat(z,c):
	return tf.concat([z,c],1)

class InfoGAN():
	def __init__(self, generator, discriminator, classifier, data):
                self.generator = generator
                self.discriminator = discriminator
                self.classifier = classifier
                self.data = data

		# data
                self.z_dim = self.data.z_dim
		#self.c_dim = self.data.y_dim # condition
                self.c_dim=20
                self.size = self.data.size
                self.channel = self.data.channel

                self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
                self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
                self.c = tf.placeholder(tf.float32, shape=[None, self.c_dim])

		# nets
		# G
                self.G_sample = self.generator(concat(self.z, self.c))
		# D
                self.D_real, self.Q_real = self.discriminator(self.X)
                self.D_fake, self.Q_fake = self.discriminator(self.G_sample, reuse = True)
                # self.G_real =self.generator(concat(self.z,self.Q_real), reuse = True)
		# Q
		# self.Q_fake = self.classifier(self.G_sample)
		
		# loss
                self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
                self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
                # fake_discri=tf.sigmoid(self.D_fake)
                # cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Q_fake[:,:10], labels=self.c[:,:10]))
                # continuals_out=self.Q_fake[:,10:self.c_dim]
                # continuals_real=self.c[:,10:self.c_dim]
                continuals_out=self.Q_fake[:,:self.c_dim]
                continuals_real=self.c[:,:self.c_dim]
                std=tf.reduce_mean(tf.abs(self.Q_fake[:,self.c_dim:]),axis=0)
                print(std.shape)
                TINY=0.0000001
                epsilon=tf.identity(tf.reduce_mean(tf.square(continuals_out-continuals_real),axis=0)/(2.*tf.square(std)+TINY)+tf.log(std+TINY))
                print(epsilon.shape)
                epsilon=1./epsilon
                continuals_loss=tf.reduce_mean(epsilon)
                self.Q_loss=continuals_loss
		# solver
                self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator.vars)
                self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator.vars)
                self.Q_solver = tf.train.AdamOptimizer().minimize(self.Q_loss, var_list=self.generator.vars + self.discriminator.vars)
                tf.summary.scalar('D_loss',self.D_loss)
                tf.summary.scalar('G_loss',self.G_loss)
                tf.summary.scalar('Q_loss',self.Q_loss)
                tf.summary.histogram('Q_fake',self.Q_fake)
                # print(self.discriminator.type)
                # print(typeof(self.discriminator))
                for var in self.discriminator.vars:
                        tf.summary.histogram(var.name,var)
                for var in self.generator.vars:
                        tf.summary.histogram(var.name,var)
                self.saver = tf.train.Saver()
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
                # gpu_options = tf.GPUOptions()
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 512):
		fig_count = 0
		train_writer = tf.summary.FileWriter(sample_dir + '/train',self.sess.graph)
		self.sess.run(tf.global_variables_initializer())
		merged = tf.summary.merge_all()
		for epoch in range(training_epoches):
			X_b, _= self.data(batch_size)
			z_b = sample_z(batch_size, self.z_dim)
			c_b = sample_c(batch_size, self.c_dim)
			# update D
			k = 1
			for _ in range(k):
				summary,_=self.sess.run(
					[merged,self.D_solver],
					feed_dict={self.X: X_b, self.z: z_b, self.c: c_b}
				)
			# update G
			
			for _ in range(k*2):
				self.sess.run(
					self.G_solver,
					feed_dict={self.z: z_b, self.c: c_b}
				)
			# update Q	
			if epoch>20000 :
				self.sess.run(
					self.Q_solver,
					feed_dict={self.z: z_b, self.c: c_b}
				)

			
			train_writer.add_summary(summary,epoch)
			# train_writer.add_summary(summary[1],epoch)
			
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr = self.sess.run(
						self.D_loss,
            			feed_dict={self.X: X_b, self.z: z_b, self.c: c_b})
				G_loss_curr, Q_loss_curr = self.sess.run(
						[self.G_loss, self.Q_loss],
						feed_dict={self.z: z_b, self.c: c_b})
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Q_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, Q_loss_curr))

				if epoch % 1000 == 0:
					z_s = sample_z(16, self.z_dim)
					c_s = sample_c(16, self.c_dim, fig_count%10)
					samples = self.sess.run(self.G_sample, feed_dict={self.c: c_s, self.z: z_s})

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
					fig_count += 1
					plt.close(fig)

				if epoch % 2000 == 0:
					self.saver.save(self.sess, os.path.join(ckpt_dir, "infogan.9labels.advance.2.ckpt"))




if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	# save generated images
	sample_dir = 'Samples/mnist_infogan_conv_with_9labels_2'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_conv_mnist()
	discriminator = D_conv_mnist_expanded(40)
	classifier = C_conv_mnist_diff_scale(9)
	data = mnist()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
	# run
	infogan = InfoGAN(generator, discriminator, classifier, data)
	infogan.train(sample_dir)

