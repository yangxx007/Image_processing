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
		self.c_dim=9
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
		self.G_real =self.generator(concat(self.z,self.Q_real), reuse = True)
		# Q
		# self.Q_fake = self.classifier(self.G_sample)
		
		# loss
		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
		self.G_loss = - tf.reduce_mean(self.D_fake)

		self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=self.generator.vars)
		# self.Q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.slice(self.Q_fake,[0,0],[self.Q_fake.shape[0],10]), labels=tf.slice(self.c,[0,0],[self.c.shape[0],10]))+tf.nn.l2_loss(tf.slice(self.Q_fake,[0,10],[self.Q_fake.shape[0],2])-tf.slice(self.c,[0,10],[self.c.shape[0],2])))
		# cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Q_fake[:,:10], labels=self.c[:,:10]))
		# +tf.nn.l2_loss(self.c[:self.Q_fake.shape[0],10:12]-self.Q_fake[:self.Q_fake.shape[0],10:12]))
		TINY=0.000001
		# continuals_out=self.Q_fake[:,10:self.c_dim]
		# continuals_real=self.c[:,10:self.c_dim]
		self.Q_loss={}
		continuals_out=self.Q_fake
		continuals_real=self.c
		std=tf.ones_like(continuals_out)
		epsilon=tf.identity(tf.square(continuals_out-continuals_real)/(tf.abs(continuals_out)+std))
		# q_loss_2=tf.reduce_sum(-0.5*np.log(2*np.pi)-tf.log(std+TINY)-0.5*tf.square(epsilon))
		continual_loss=tf.reduce_mean(epsilon)
		self.Q_loss[0]=continual_loss
		continuals_out=self.Q_fake[:,3:9]
		continuals_real=self.c[:,3:9]
		std=tf.ones_like(continuals_out)
		epsilon=tf.identity(tf.square(continuals_out-continuals_real)/(tf.abs(continuals_out)+std))
		continual_loss_1=tf.reduce_mean(epsilon)
		self.Q_loss[1]=continual_loss_1
		continuals_out=self.Q_fake[:,6:9]
		continuals_real=self.c[:,6:9]
		std=tf.ones_like(continuals_out)
		epsilon=tf.identity(tf.square(continuals_out-continuals_real)/(tf.abs(continuals_out)+std))
		continual_loss_2=tf.reduce_mean(epsilon)
		self.Q_loss[2]=continual_loss_2
		self.consistentloss=tf.reduce_mean(tf.square(self.X-self.G_real)/(tf.abs(self.X)+TINY))
		
		# solver
		self.Q_solver ={}
		# self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator.vars)
		# self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator.vars)
		self.Q_solver[0] = tf.train.AdamOptimizer().minimize(self.Q_loss[2], var_list=self.generator.vars + self.discriminator.vars)
		self.Q_solver[1] = tf.train.AdamOptimizer().minimize(self.Q_loss[1], var_list=self.generator.vars + self.discriminator.vars)
		self.Q_solver[2] = tf.train.AdamOptimizer().minimize(self.Q_loss[0], var_list=self.generator.vars + self.discriminator.vars)
		self.consistent_solver=tf.train.AdamOptimizer().minimize(self.consistentloss,var_list=self.generator.vars + self.discriminator.vars)
		tf.summary.scalar('D_loss',self.D_loss)
		tf.summary.scalar('G_loss',self.G_loss)
		tf.summary.scalar('Q_loss_0',self.Q_loss[0])
		tf.summary.scalar('Q_loss_1',self.Q_loss[1])
		tf.summary.scalar('Q_loss_2',self.Q_loss[2])
		tf.summary.scalar('consistent_loss',self.consistentloss)
		# tf.summary.scalar('continus_loss',continual_loss)
		self.saver = tf.train.Saver()
		# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
		gpu_options = tf.GPUOptions()
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 64):
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
				if epoch>400000 :
					summary,_,_=self.sess.run(
					[merged,self.D_solver,self.consistent_solver],
					feed_dict={self.X: X_b, self.z: z_b, self.c: c_b})
					continue
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
			if epoch>0 and epoch<200000:
				self.sess.run(
					self.Q_solver[0],
					feed_dict={self.z: z_b, self.c: c_b}
				)
			if epoch>200000 and epoch<400000:
				self.sess.run(
					self.Q_solver[1],
					feed_dict={self.z: z_b, self.c: c_b}
				)
			if epoch>400000 :
				self.sess.run(
					self.Q_solver[2],
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
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Q_loss[0]: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, Q_loss_curr[0]))

				if epoch % 1000 == 0:
					z_s = sample_z(16, self.z_dim)
					c_s = sample_c(16, self.c_dim, fig_count%10)
					samples = self.sess.run(self.G_sample, feed_dict={self.c: c_s, self.z: z_s})

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
					fig_count += 1
					plt.close(fig)

				if epoch % 2000 == 0:
					self.saver.save(self.sess, os.path.join(ckpt_dir, "infogan.diffScale_9_advance.consistentloss.ckpt"))




if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	# save generated images
	sample_dir = 'Samples/mnist_infogan_conv_diffscale_9_advance_consistentloss'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_conv_mnist()
	discriminator = D_conv_mnist_diff_scale(9)
	classifier = C_conv_mnist_diff_scale(9)
	data = mnist()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
	# run
	infogan = InfoGAN(generator, discriminator, classifier, data)
	infogan.train(sample_dir)

