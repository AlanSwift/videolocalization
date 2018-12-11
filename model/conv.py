import torch
import torch.nn as nn

class conv_layer(nn.Module):
	def __init__(self, inputs_dim, hidden, kwidth, dropout):
		'''
			hidden: [24, 48, 96]
			kwidth: [3, 3, 3]

		'''
		super(conv_layer, self).__init__()
		assert len(hidden) == len(kwidth)
		layers_list = []
		self.layers_num = len(hidden)
		self.moduleList = nn.ModuleList()
		self.resModule = nn.ModuleList()
		self.batchNormalizeList = nn.ModuleList()
		self.fc = nn.Linear(in_features=inputs_dim, out_features=hidden[0], bias=True)

		for layer_idx in range(len(hidden)):
			nin = hidden[layer_idx] if layer_idx == 0 else hidden[layer_idx - 1]
			nout = hidden[layer_idx]
			print("...", nin, nout)
			if nin != nout:
				input_dim = inputs_dim if layer_idx == 0 else nin
				self.resModule.append(nn.Sequential(self.linear_mapping(input_dim=input_dim, out_dim=nout,
									dropout=dropout['src'])))
			else:
				self.resModule.append(None)

			layers_list.append(nn.Dropout(dropout['src']))
			layers_list.append(self.conv1d_with_bias(input_dim=nout, output_dim=nout*2, 
								ksize=kwidth[layer_idx], dropout=dropout['hid']))
			self.moduleList.append(nn.Sequential(*layers_list))
			layers_list = []
			self.batchNormalizeList.append(nn.BatchNorm1d(nout))



	def conv1d_with_bias(self, input_dim, output_dim, ksize, dropout=1.0):
		'''
			w-k+2p //stride + 1 == w
		'''
		return nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=ksize,
						stride=1, padding=(ksize-1)//2, bias=True)


	def linear_mapping(self, input_dim, out_dim, dropout: int):
		return [nn.Linear(in_features = input_dim, out_features = out_dim,
							bias = True), nn.Dropout(dropout)]

	def gated_linear_units(self, inputs):
		input_shape = list(inputs.size())
		assert len(input_shape) == 3
		input_pass = inputs[:, :, 0:input_shape[-1]//2]
		input_gate = inputs[:, :, input_shape[-1]//2:]
		input_gate = nn.Sigmoid()(input_gate)
		return torch.mul(input_pass, input_gate)

	def forward(self, inputs):
		next_layer = self.fc(inputs)
		for i in range(self.layers_num):
			print(i)
			if self.resModule[i] is None:
				res_inputs = next_layer
			else:
				res_inputs = self.resModule[i](next_layer)
			next_layer = next_layer.transpose(1, 2)
			next_layer = self.moduleList[i](next_layer)
			next_layer = next_layer.transpose(1, 2)
			next_layer = self.gated_linear_units(next_layer)
			next_layer = (next_layer + res_inputs) * torch.sqrt(torch.tensor(0.5).cuda())
			next_layer = next_layer.transpose(1, 2)
			next_layer = self.batchNormalizeList[i](next_layer).transpose(1, 2)
		return next_layer


