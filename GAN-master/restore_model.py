import tensorflow as tf
from infogan_conv_without_share import *
import numpy
def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])
def sample_c_2(m, n, ind=-1):
	# c = np.zeros([m,n])
	c= np.zeros([m,n])
	for i in range(m):
		if ind<0:
			ind = np.random.randint(10)
		# c[i,i%10] = 1.
		for j in range(0,n):
			c[i,j]=np.random.uniform(-2.,2.)
	return c
def sample_c(m, n,init,ind=-1):
	# c = np.zeros([m,n])
	c= np.zeros([m,n])
	for i in range(m):
		if ind<0:
			ind = np.random.randint(10)
		c[i,i%10] = 1.
		c[i,10]=0
		c[i,11]=np.random.uniform(-2,2)
	return c
meta_path = './ckpt/infogan.diffScale_9_advance.consistentloss.ckpt.meta'  
model_path = './ckpt/infogan.diffScale_9_advance.consistentloss.ckpt'  
image_name='infogan_diffscale_'
sample_dir='./test_result/advance'
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
batch_size=256
z_dim=100
c_dim=9
generator = G_conv_mnist()
discriminator = D_conv_mnist_diff_scale(9)
classifier = C_conv_mnist_diff_scale(9)
data = mnist()
# run
model = InfoGAN(generator, discriminator, classifier, data)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(meta_path) # 导入图
    init_l = tf.global_variables_initializer()
    sess.run(init_l)  
    saver.restore(sess, model_path)  
    z_b = sample_z(batch_size, z_dim) 
    #saver.restore(sess, tf.train.latest_checkpoint('./ckpt/')) # 导入变量值  
    # graph = tf.get_default_graph()
    for i in range (0,16):
        # z_b = sample_z(batch_size, z_dim) 
        for k in range(0,3):
            for j in range(0,16):
                c_b = sample_c_2(batch_size, c_dim,j*0.2)
                c_b[i]=i/16.*4.-2
                c_b[i][3*k]=j/16.*4.-2.
                c_b[i][3*k+1]=i/16.*4.-2
                c_b[i][3*k+2]=i/16.*4.-2
                tmp=sess.run(model.G_sample, feed_dict={model.z:z_b,model.c:c_b})
                if j!=0:
                    sample=np.append(sample,tmp[i:i+1],0)
                else:
                    sample=tmp[i:i+1]
            fig = model.data.data2fig(sample)
            plt.savefig('{}/{}_{}.png'.format(sample_dir, image_name+"_advance_"+str(k+3)+str(i).zfill(3), str(1)), bbox_inches='tight')
            plt.close(fig)
    #print(sess.run(graph.get_tensor_by_name('logits_classifier/weights:0'))) # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重  