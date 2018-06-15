import numpy as np
import scipy.io as sio # load matlab file *.mat
import tensorflow as tf
import scipy
import os

store_path = "./store/"
content_path = './style/rain_princess.jpg'
style_path = './style/udnie.jpg'

image_height = 800#224#600
image_width = 800#224#800
learning_rate = 10

ratio = 0.00001 # alpha/beta #row ratio == big beta
beta = 10
alpha = beta * ratio
RGB_process = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))



#http://www.vlfeat.org/matconvnet/pretrained/  에서 imagenet-vgg-verydeep-19 파일다운로드.
pretrained = sio.loadmat('./imagenet-vgg-verydeep-19.mat')
vgg19 = pretrained['layers']


#사진파일 읽어오기
def read_image(path):
	img = tf.keras.preprocessing.image.load_img(path, target_size=(image_height, image_width)) 
	img = tf.keras.preprocessing.image.img_to_array(img) #img = height,width,3
	img = np.array([img])
	img -= RGB_process
	return img


#가지고 있는 vgg19 이름 출력함.
#conv1_1, relu1_1, conv1_2, .... prob 
def show_vgg19_info(vgg19):
	for i in range(len(vgg19[0])): 
		print(vgg19[0][i][0][0][0][0])



#dictionary 생성함.# {conv1_1:0, relu1_1:1, ... , fc8:41, prob:42}
def set_vgg19_info_dict(vgg19):
	dic = {}
	for i in range(len(vgg19[0])): 
		dic[vgg19[0][i][0][0][0][0]] = i
	return dic



def conv2d(vgg19, input_, layer): #name은 show_vgg19_info 실행하면 나오는 이름 기반임.
	weight = tf.constant(vgg19[0][layer][0][0][2][0][0]) #wight
	bias = np.transpose(vgg19[0][layer][0][0][2][0][1]) # bias (x, 1) 꼴이여서 transpose 해줘야됨.
	bias = tf.constant(bias)
	return tf.nn.conv2d(input_, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
	


def vgg19_network(vgg19, input_, layer_info):
	conv1_1 = conv2d(vgg19, input_, layer_info['conv1_1'])
	relu1_1 = tf.nn.relu(conv1_1)
	conv1_2 = conv2d(vgg19, relu1_1, layer_info['conv1_2'])
	relu1_2 = tf.nn.relu(conv1_2)
	pool1 = tf.nn.avg_pool(relu1_2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	conv2_1 = conv2d(vgg19, pool1, layer_info['conv2_1'])
	relu2_1 = tf.nn.relu(conv2_1)
	conv2_2 = conv2d(vgg19, relu2_1, layer_info['conv2_2'])
	relu2_2 = tf.nn.relu(conv2_2)
	pool2 = tf.nn.avg_pool(relu2_2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	conv3_1 = conv2d(vgg19, pool2, layer_info['conv3_1'])
	relu3_1 = tf.nn.relu(conv3_1)
	conv3_2 = conv2d(vgg19, relu3_1, layer_info['conv3_2'])
	relu3_2 = tf.nn.relu(conv3_2)
	conv3_3 = conv2d(vgg19, relu3_2, layer_info['conv3_3'])
	relu3_3 = tf.nn.relu(conv3_3)
	conv3_4 = conv2d(vgg19, relu3_3, layer_info['conv3_4'])
	relu3_4 = tf.nn.relu(conv3_4)
	pool3 = tf.nn.avg_pool(relu3_4, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	conv4_1 = conv2d(vgg19, pool3, layer_info['conv4_1'])
	relu4_1 = tf.nn.relu(conv4_1)
	conv4_2 = conv2d(vgg19, relu4_1, layer_info['conv4_2'])
	relu4_2 = tf.nn.relu(conv4_2)
	conv4_3 = conv2d(vgg19, relu4_2, layer_info['conv4_3'])
	relu4_3 = tf.nn.relu(conv4_3)
	conv4_4 = conv2d(vgg19, relu4_3, layer_info['conv4_4'])
	relu4_4 = tf.nn.relu(conv4_4)
	pool4 = tf.nn.avg_pool(relu4_4, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	conv5_1 = conv2d(vgg19, pool4, layer_info['conv5_1'])
	relu5_1 = tf.nn.relu(conv5_1)
	conv5_2 = conv2d(vgg19, relu5_1, layer_info['conv5_2'])
	relu5_2 = tf.nn.relu(conv5_2)
	conv5_3 = conv2d(vgg19, relu5_2, layer_info['conv5_3'])
	relu5_3 = tf.nn.relu(conv5_3)
	conv5_4 = conv2d(vgg19, relu5_3, layer_info['conv5_4'])
	relu5_4 = tf.nn.relu(conv5_4)
	pool5 = tf.nn.avg_pool(relu5_4, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 이 이후로는 안쓰이므로 구축 안함.

	content = {'conv4_2':conv4_2}
	style = {'conv1_1':conv1_1, 'conv2_1':conv2_1, 'conv3_1':conv3_1, 'conv4_1':conv4_1, 'conv5_1':conv5_1}
	
	return content, style



def calc_content_loss(noise_content, content):
	loss = tf.reduce_sum(tf.square(noise_content['conv4_2'] - content['conv4_2']))/2 #원본 이미지의 conv 첫레이어 결과랑, 노이즈로 만든 conv 첫레이어 결과 차이
	return loss



def gram_matrix(data, M, N):
	# M: height*width, N: filter, data.shape:(1, height, width, N)
	flatten = tf.transpose(data, [3, 1, 2, 0]) # shape:(N, height, width, 1) 
	flatten = tf.reshape(flatten, (N, M)) # shape:(N, M)
	gram = tf.matmul(flatten, tf.transpose(flatten))
	return gram



def calc_style_loss(noise_style, style):
	grams = []
	for i in noise_style: #noise_style: {'conv1_1':conv1_1, 'conv2_1':conv2_1, 'conv3_1':conv3_1, 'conv4_1':conv4_1, 'conv5_1':conv5_1}
		shape = noise_style[i].get_shape()
		M = int(shape[1]*shape[2]) #height*width
		N = int(shape[-1]) #filter
		
		#gram 계산.		
		noise_gram_matrix = gram_matrix(noise_style[i], M, N) #노이즈 이미지의 그램 계산
		style_gram_matrix = gram_matrix(style[i], M, N) #스타일 이미지의 그램 계산
		grams.append(((tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix)) / (4*N*N*M*M) )) * 0.2) #논문의 수식, 0.2:W_l
		
	style_loss = tf.reduce_sum(grams)
	return style_loss



def neural_style_transfer():
	if not os.path.exists(store_path):
		os.makedirs(store_path)

	for epoch in range(1000):
		cost_, _ = sess.run([cost, minimize])

		if epoch % 50 == 0:
			print("epoch: ",epoch, " cost: ",cost_)
			#이미지 후처리.
			noise_ = sess.run(noise)
			noise_ += RGB_process
			noise_ = np.clip(noise_, 0, 255).astype(np.uint8)
			noise_ = np.reshape(noise_, [image_height,image_width,3])

			#이미지 저장.
			scipy.misc.imsave(store_path+str(epoch)+'.jpg', noise_)



#show_vgg19_info(vgg19) #('./imagenet-vgg-verydeep-19.mat') 의 layer 정보 보여줌.
layer_info = set_vgg19_info_dict(vgg19) # {conv1_1:0, relu1_1:1, ... , fc8:41, prob:42}, vgg19_network 계산시 활용됨.


#content 이미지 읽고 conv 값들 계산.
with tf.name_scope("content"):
	content_ = read_image(content_path)
	content, _ = vgg19_network(vgg19, tf.constant(content_), layer_info)


#style 이미지 읽고 conv 값들 계산.
with tf.name_scope("style"):
	style_ = read_image(style_path)
	_, style = vgg19_network(vgg19, tf.constant(style_), layer_info)
	

#noise의 초기값을 content로 세팅하고 conv 값들 계산.
with tf.variable_scope("noise"):
	noise = tf.Variable( tf.constant(content_)) #noise init content
	noise_content, noise_style = vgg19_network(vgg19, noise, layer_info)


#content cost 계산
content_cost = calc_content_loss(noise_content, content)

#style cost 계산
style_cost = calc_style_loss(noise_style, style) 

#weight인 noise가 backpropagation을 통해 바뀌면 style_cost + content_cost가 점점 줄어들음
cost = alpha*content_cost + beta*style_cost

#논문과는 다르게 Adam Optimizer 사용함. (구현상의 이유)
optimizer = tf.train.AdamOptimizer(learning_rate)
minimize = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


#run
neural_style_transfer()
