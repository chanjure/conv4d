import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class Conv4d(nn.Module):
	"""
	4D Convolution layer
	"""

	def __init__(self, ker_shape, in_ch, out_ch, bias=False):
		super(Conv4d, self).__init__()
		self.k = nn.Parameter(tc.empty((in_ch,out_ch)+ker_shape))
		if bias:
			self.bias = nn.Parameter(tc.empty(ker_shape+(out_ch,)))
		else:
			self.register_parameter('bias', None)

		nn.init.kaiming_uniform_(self.k)
		if self.bias is not None:
			nn.init.kaiming_uniform_(self.bias)

	def forward(self,x):
		"""
		input = [n, in_ch, in_dim1, in_dim2, in_dim3, in_dim4]
		kernel = [in_ch, out_ch, ker_dim1, ker_dim2, ker_dim3, ker_dim4]
		output = [n, out_ch, out_dim1, out_dim2, out_dim3, out_dim4]
		"""
		if x.shape[1] != self.k.shape[0]:
			raise ValueError('Check input channel number input:%d kernel:%d'%(x.shape[1], self.k.shape[0]))
		
		# str_shape = [n, in_ch, ker_dim, out_dim]
		# out_dim = input_dim - ker_dim + 1
		str_shape = (x.shape[0],
				x.shape[1],
				self.k.shape[2],
				self.k.shape[3],
				self.k.shape[4],
				self.k.shape[5],
				x.shape[2] - self.k.shape[2] + 1,
				x.shape[3] - self.k.shape[3] + 1,
				x.shape[4] - self.k.shape[4] + 1,
				x.shape[5] - self.k.shape[5] + 1)

		# x_strides = [n, in_ch, in_dim, strides=in_dim]
		x_strides = (x.stride()[0],
				x.stride()[1],
				x.stride()[2],
				x.stride()[3],
				x.stride()[4],
				x.stride()[5],
				x.stride()[2],
				x.stride()[3],
				x.stride()[4],
				x.stride()[5])

		x = tc.as_strided(x,
				size=str_shape,
				stride=x_strides
				)

		return tc.tensordot(x, self.k, dims=([1,2,3,4,5],[0,2,3,4,5])).permute(0,5,1,2,3,4)

class MaxPool4d(nn.Module):
	"""
	MaxPool4d
	"""
	def __init__(self, ker_shape, stride):
		super(MaxPool4d, self).__init__()
		self.ker_shape = ker_shape
		self.stride = stride

	def forward(self, x):
		# input = [n, in_ch, in_dim1, in_dim2, in_dim3, in_dim4]
		# kernel = [ker_dim1, ker_dim2, ker_dim3, ker_dim4]
		# output = [n, in_ch, out_dim1, out_dim2, out_dim3, out_dim4]

		# str_shape = [n, inc, ker_dim, out_dim]
		str_shape = (x.shape[0],
				x.shape[1],
				self.ker_shape[0],
				self.ker_shape[1],
				self.ker_shape[2],
				self.ker_shape[3],
				int((x.shape[2] - self.ker_shape[0])/self.stride + 1),
				int((x.shape[3] - self.ker_shape[1])/self.stride + 1),
				int((x.shape[4] - self.ker_shape[2])/self.stride + 1),
				int((x.shape[5] - self.ker_shape[3])/self.stride + 1))

		# x_strides = [n, inc, in_dim, in_stride=in_dim]
		x_strides = (x.stride()[0],
				x.stride()[1],
				x.stride()[2],
				x.stride()[3],
				x.stride()[4],
				x.stride()[5],
				x.stride()[2],
				x.stride()[3],
				x.stride()[4],
				x.stride()[5])

		x = tc.as_strided(x,
				size=str_shape,
				stride=x_strides)

		return tc.amax(x, dim=(2,3,4,5))

class AvgPool4d(nn.Module):
	"""
	AvgPool4d

	Not tested
	"""
	def __init__(self, ker_shape, stride):
		super(AvgPool4d, self).__init__()
		self.ker_shape = ker_shape
		self.stride = stride

	def forward(self, x):
		# input = [n, in_ch, in_dim1, in_dim2, in_dim3, in_dim4]
		# kernel = [ker_dim1, ker_dim2, ker_dim3, ker_dim4]
		# output = [n, in_ch, out_dim1, out_dim2, out_dim3, out_dim4]

		# str_shape = [n, inc, ker_dim, out_dim]
		str_shape = (x.shape[0],
				x.shape[1],
				self.ker_shape[0],
				self.ker_shape[1],
				self.ker_shape[2],
				self.ker_shape[3],
				int((x.shape[2] - self.ker_shape[0])/self.stride + 1),
				int((x.shape[3] - self.ker_shape[1])/self.stride + 1),
				int((x.shape[4] - self.ker_shape[2])/self.stride + 1),
				int((x.shape[5] - self.ker_shape[3])/self.stride + 1))

		# x_strides = [n, inc, in_dim, in_stride=in_dim]
		x_strides = (x.stride()[0],
				x.stride()[1],
				x.stride()[2],
				x.stride()[3],
				x.stride()[4],
				x.stride()[5],
				x.stride()[2],
				x.stride()[3],
				x.stride()[4],
				x.stride()[5])

		x = tc.as_strided(x,
				size=str_shape,
				stride=x_strides)

		return tc.mean(x, dim=(2,3,4,5))
