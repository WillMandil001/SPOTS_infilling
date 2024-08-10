import math
import yaml
import torch
import inspect
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import Tensor
from types import SimpleNamespace
from dataclasses import dataclass
from transformers import AutoTokenizer


class LayerNorm(nn.Module):
	""" LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
	def __init__(self, ndim, bias):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(ndim))
		self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

	def forward(self, input):
		return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		self.mask = config.mask
		self.config = config
		self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # key, query, value projections for all heads, but in a batch
		self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)      # output projection
		self.attn_dropout = nn.Dropout(config.dropout)                               # regularization
		self.resid_dropout = nn.Dropout(config.dropout)
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		self.dropout = config.dropout
		# flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
		self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
		if not self.flash:
			print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
			# causal mask to ensure that attention is only applied to the left in the input sequence
			self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

		if self.mask:
			if config.action == False and config.tactile == False:
				self.maskarray = torch.zeros(config.context_length * config.patches_per_frame, config.context_length * config.patches_per_frame)
				for i in range(config.context_length):
					start_row, end_row = i * config.patches_per_frame, (i + 1) * config.patches_per_frame
					self.maskarray[start_row:end_row, :end_row] = 1
				self.maskarray = self.maskarray.to(config.device)
				self.maskarray = self.maskarray.type(torch.bool)

			if config.action == True and config.tactile == False:
				self.maskimage = torch.zeros(config.context_length * config.patches_per_frame, config.context_length * config.patches_per_frame)
				for i in range(config.context_length):
					start_row, end_row = i * config.patches_per_frame, (i + 1) * config.patches_per_frame
					self.maskimage[start_row:end_row, :end_row] = 1

				# action queries - image keys
				image_maskarray = torch.zeros(config.context_length+1, config.context_length * config.patches_per_frame)
				for i in range(config.context_length+1):
					start_row, end_row = i, (i + 1)
					image_maskarray[start_row:end_row, :end_row * config.patches_per_frame] = 1

				# action queries - action keys
				action_maskarray = torch.zeros(config.context_length+1, config.context_length+1)
				for i in range(0, config.context_length+1):
					start_row, end_row = i, (i + 2)
					action_maskarray[start_row:end_row, :end_row] = 1

				# image queries - action keys  (160, 11)
				self.maskarray_action_X = torch.zeros(config.context_length * config.patches_per_frame, config.context_length+1)
				for i in range(config.context_length):
					start_row, end_row = i, (i + 2)
					self.maskarray_action_X[start_row * config.patches_per_frame:end_row * config.patches_per_frame, :end_row] = 1

				self.maskarray_action_Y = torch.cat([action_maskarray, image_maskarray], axis = 1)

				self.maskarray = torch.cat([self.maskarray_action_X, self.maskimage], axis = 1)
				self.maskarray = torch.cat([self.maskarray_action_Y, self.maskarray], axis = 0)

				self.maskarray = self.maskarray.to(config.device)
				self.maskarray = self.maskarray.type(torch.bool)

			if config.action == True and config.tactile == True:
				context_length = config.context_length
				num_frames = config.num_frames
				patches_per_frame = config.patches_per_frame
				patches_per_tactile_frame = config.patches_per_tactile_frame

				maskimage = torch.zeros(context_length * patches_per_frame, context_length * patches_per_frame)
				for i in range(context_length):
					start_row, end_row = i * patches_per_frame, (i + 1) * patches_per_frame
					maskimage[start_row:end_row, :end_row] = 1
				image_maskarray = maskimage.type(torch.bool)
				image_maskarray_copy = image_maskarray.clone()
				# 2: image - tactile
				tactile_maskarray = torch.zeros(context_length * patches_per_frame, context_length * patches_per_tactile_frame)
				for i in range(context_length):
					start_row, end_row = i, (i + 1)
					tactile_maskarray[start_row * patches_per_frame:end_row * patches_per_frame, :end_row*patches_per_tactile_frame] = 1
				tactile_maskarray = tactile_maskarray.type(torch.bool)

				# 3: image - action
				# action_maskarray = torch.ones(context_length * patches_per_frame, num_frames)
				action_maskarray = torch.zeros(context_length * patches_per_frame, context_length+1)
				for i in range(context_length):
					start_row, end_row = i, (i + 2)
					action_maskarray[start_row * patches_per_frame:end_row * patches_per_frame, :end_row] = 1

				mask_array1 = torch.cat([action_maskarray, tactile_maskarray, image_maskarray], axis = 1)

				##### tactile queries
				# 4. tactile - images
				image_maskarray = torch.zeros(context_length * patches_per_tactile_frame, context_length * patches_per_frame)
				for i in range(context_length):
					start_row, end_row = i, (i + 1)
					image_maskarray[start_row*patches_per_tactile_frame:end_row*patches_per_tactile_frame, :end_row * patches_per_frame] = 1
				image_maskarray = image_maskarray.type(torch.bool)

				# 5. tactile - tactile
				tactile_maskarray = torch.zeros(context_length * patches_per_tactile_frame, context_length * patches_per_tactile_frame)
				for i in range(context_length):
					start_row, end_row = i, (i + 1)
					tactile_maskarray[start_row * patches_per_tactile_frame:end_row * patches_per_tactile_frame, :end_row * patches_per_tactile_frame] = 1
				tactile_maskarray = tactile_maskarray.type(torch.bool)

				# 6. tactile queries - action keys
				action_maskarray = torch.zeros(context_length * patches_per_tactile_frame, context_length + 1)
				for i in range(context_length):
					start_row, end_row = i, (i + 2)
					action_maskarray[start_row * patches_per_tactile_frame:end_row * patches_per_tactile_frame, :end_row] = 1

				mask_array2 = torch.cat([action_maskarray, tactile_maskarray, image_maskarray], axis = 1)

				##### action queries
				# 7 action - image
				image_maskarray = torch.zeros(context_length+1, context_length * patches_per_frame)
				for i in range(context_length+1):
					start_row, end_row = i, (i + 1)
					image_maskarray[start_row:end_row, :end_row * patches_per_frame] = 1
				image_maskarray = image_maskarray.type(torch.bool)

				# 8 action - tactile
				tactile_maskarray = torch.zeros(context_length+1, context_length*patches_per_tactile_frame)
				for i in range(context_length+1):
					start_row, end_row = i, (i + 1)
					tactile_maskarray[start_row:end_row, :end_row*patches_per_tactile_frame] = 1
				tactile_maskarray = tactile_maskarray.type(torch.bool)

				# 9 action - action
				actionaction_maskarray = torch.zeros(context_length+1, context_length+1)
				for i in range(0, context_length+1):
					start_row, end_row = i, (i + 2)
					actionaction_maskarray[start_row:end_row, :end_row] = 1

				# 10 combine all the masks
				mask_array3 = torch.cat([actionaction_maskarray, tactile_maskarray, image_maskarray], axis = 1)

				self.maskarray = torch.cat([mask_array3, mask_array2, mask_array1], axis = 0)
				self.maskarray = self.maskarray.to(config.device)
				self.maskarray = self.maskarray.type(torch.bool)

	def viz_mask(self):
		plt.imshow(self.maskarray.numpy(), cmap='gray', extent=[0, self.maskarray.numpy().shape[1], self.maskarray.numpy().shape[0], 0])
		plt.title('Attention Mask')
		plt.xlabel('Key Positions')
		plt.ylabel('Query Positions')
		plt.colorbar()

		x_ticks = [0, 0]
		y_ticks = [0, 0]
		x_gaps_labels = []
		y_gaps_labels = []

		if self.config.action == True:
			action_x_ticks = [(i) for i in range(0, self.config.context_length+1)]
			action_y_ticks = [(i) for i in range(0, self.config.context_length+1)]
			x_ticks += action_x_ticks
			y_ticks += action_y_ticks

			x_gaps_labels += [f'Robot State {i}' for i in range(1, len(action_x_ticks) + 1)]
			y_gaps_labels += [f'Robot State {i}' for i in range(1, len(action_y_ticks) + 1)]

		if self.config.tactile == True:
			tactile_x_ticks = [(i + x_ticks[-1]) for i in range(x_ticks[-1] - x_ticks[-2], ((self.config.context_length) * self.config.patches_per_tactile_frame), self.config.patches_per_tactile_frame)]
			tactile_y_ticks = [(i + y_ticks[-1]) for i in range(y_ticks[-1] - y_ticks[-2], ((self.config.context_length) * self.config.patches_per_tactile_frame), self.config.patches_per_tactile_frame)]
			x_ticks += tactile_x_ticks
			y_ticks += tactile_y_ticks

			x_gaps_labels += [f'Tactile Patch {i}'  for i in range(1, self.config.context_length+1)]
			y_gaps_labels += [f'Tactile Patch {i}'  for i in range(1, self.config.context_length+1)]

		if self.config.image == True:
			image_x_ticks = [(i + x_ticks[-1]) for i in range(x_ticks[-1] - x_ticks[-2], ((self.config.context_length+1) * self.config.patches_per_frame), self.config.patches_per_frame)]
			image_y_ticks = [(i + y_ticks[-1]) for i in range(y_ticks[-1] - y_ticks[-2], ((self.config.context_length+1) * self.config.patches_per_frame), self.config.patches_per_frame)]
			x_ticks += image_x_ticks
			y_ticks += image_y_ticks

			x_gaps_labels += [f'Image Patch {i}' for i in range(1, self.config.context_length + 1)]
			y_gaps_labels += [f'Image Patch {i}' for i in range(1, self.config.context_length + 1)]

		# remove the first two zeros
		x_ticks = x_ticks[2:]
		y_ticks = y_ticks[2:]

		# Compute the middle points between the ticks
		x_gaps = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
		y_gaps = [(y_ticks[i] + y_ticks[i+1]) / 2 for i in range(len(y_ticks) - 1)]

		# place the labels at x_gaps but remove the actual ticks
		plt.xticks(x_gaps, x_gaps_labels, rotation=90, fontsize=6)
		plt.yticks(y_gaps, y_gaps_labels, fontsize=6)

		# draw grid lines at the x_ticks and y_ticks
		plt.grid(which='minor', color='gray', linestyle='-', linewidth=1.0)
		plt.gca().set_xticks(x_ticks, minor=True)
		plt.gca().set_yticks(y_ticks, minor=True)

		# Remove the actual tick lines for the labeled ticks
		plt.tick_params(axis='x', which='major', length=0)
		plt.tick_params(axis='y', which='major', length=0)

		return plt

	def forward(self, x):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

		# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
		if self.flash:
			# efficient attention using Flash Attention CUDA kernels
			y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
		else:
			# manual implementation of attention
			att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
			if self.mask:
				extended_mask = self.maskarray[None, None, :, :].repeat(B, self.n_head, 1, 1)
				att = att.masked_fill(extended_mask == 0, float('-inf'))
				# att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

			att = F.softmax(att, dim=-1)
			att = self.attn_dropout(att)
			y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output projection
		y = self.resid_dropout(self.c_proj(y))
		return y


class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
		self.gelu    = nn.GELU()
		self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		x = self.dropout(x)
		return x


class Block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class VPGPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.block_size is not None
		self.config = config

		self.transformer = nn.ModuleDict(dict(
			patch_and_embed = nn.Conv2d(in_channels=config.input_dim, out_channels=config.n_embd, kernel_size=(config.fh, config.fw), stride=(config.fh, config.fw), padding=0),
			wpe = nn.Embedding(config.block_size, config.n_embd),
			drop = nn.Dropout(config.dropout),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			ln_f = LayerNorm(config.n_embd, bias=config.bias),
			decode_and_depatch = nn.ConvTranspose2d(in_channels=config.n_embd, out_channels=config.input_dim, kernel_size=(config.fh, config.fw), stride=(config.fh, config.fw), padding=0),
			sig = nn.Sigmoid()
			))

		if config.action:
			self.transformer.action_embedding  = nn.Linear(config.action_dim, config.n_embd)

		if config.tactile:
			self.transformer.tactile_embedding = nn.Linear(config.tactile_dim, config.n_embd)
			self.transformer.tactile_debedding = nn.Linear(config.n_embd, config.tactile_dim)

		if self.config.load_pretrained_image_model or self.config.load_pretrained_ac_image_model:      
			pretrained_image_model_config_path = config.pretrained_config_path
			with open(pretrained_image_model_config_path, 'r') as file:  pretrained_image_model_config = yaml.load(file, Loader=yaml.FullLoader)
			pretrained_image_model_config = SimpleNamespace(**pretrained_image_model_config["model_config"]["value"])
			self.transformer["prior_model"] = VPGPT(pretrained_image_model_config)
	
		self.apply(self._init_weights)  # careful initialization
		for pn, p in self.named_parameters():
			if pn.endswith('c_proj.weight'):  # apply special scaled init to the residual projections, per GPT-2 paper
				torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

		if self.config.load_pretrained_image_model or self.config.load_pretrained_ac_image_model:
			print("!!! LOADING THE pretrained image encoder !!!")
			self.transformer.prior_model.load_state_dict(torch.load(self.config.pretrained_model_path))

		if self.config.load_pretrained_image_tokenizer:
			print("!!! LOADING THE pretrained image tokenizer !!!")
			pretrained_state_dict = torch.load(self.config.pretrained_model_path)
			patch_and_embed_state_dict = {k.replace('transformer.patch_and_embed.', ''): v for k, v in pretrained_state_dict.items() if 'transformer.patch_and_embed.' in k}
			self.transformer.patch_and_embed.load_state_dict(patch_and_embed_state_dict)

		if self.config.load_pretrained_image_decoder:    
			print("!!! LOADING THE pretrained image decoder   !!!")
			pretrained_state_dict = torch.load(self.config.pretrained_model_path)
			decode_and_depatch_state_dict = {k.replace('transformer.decode_and_depatch.', ''): v for k, v in pretrained_state_dict.items() if 'transformer.decode_and_depatch.' in k}
			self.transformer.decode_and_depatch.load_state_dict(decode_and_depatch_state_dict)

	def get_attention_mask(self):
		return self.transformer.h[0].attn.viz_mask()

	def forward(self, idx, targets=None, actions=None, tactiles=None, tactile_targets=None, test=False):
		device = idx.device

		# pre-process the input if we are using a pretrained model for the image tokenizer
		if self.config.load_pretrained_image_model or self.config.load_pretrained_ac_image_model:
			new_idx, _, _, _, _ = self.transformer.prior_model(idx, targets, actions, tactiles, tactile_targets)
			tok_emb = new_idx.view(-1, self.config.input_dim, self.config.H, self.config.W)
		else:
			tok_emb = idx.view(-1, self.config.input_dim, self.config.H, self.config.W)

		tok_emb = self.transformer.patch_and_embed(tok_emb)  # shape (b, n_embd, t, t)

		patch_size = tok_emb.shape[2]
		tok_emb = tok_emb.flatten(2)
		tok_emb = tok_emb.view(-1, self.config.context_length, tok_emb.shape[1], tok_emb.shape[2])  # seperate/split apart the batch and the time_step dims 
		tok_emb = tok_emb.transpose(2, 3)															# flip the enc_dim from being the last axis to the second last (allows us to combine in the next step)
		tok_emb = tok_emb.reshape(tok_emb.shape[0], -1, tok_emb.shape[3])							# flatten the time_steps and the patches into one long sequence

		if self.config.tactile:
			tactile_emb = self.transformer.tactile_embedding(tactiles)
			tok_emb = torch.cat((tactile_emb, tok_emb), 1)

		if self.config.action:
			action_emb = self.transformer.action_embedding(actions)
			tok_emb = torch.cat((action_emb, tok_emb), 1)

		n, t, c = tok_emb.shape
		assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}, check you have added an extra bit for the classification token"

		pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
		pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
		x = self.transformer.drop(tok_emb + pos_emb)

		for block in self.transformer.h:
			x = block(x)
		x = self.transformer.ln_f(x)

		if self.config.action and not self.config.tactile:
			action, x = torch.split(x, [self.config.context_length+1, self.config.context_length * self.config.patches_per_frame], dim=1)
		elif self.config.tactile and not self.config.action:
			x_tactile, x  = torch.split(x, [self.config.context_length + 1, self.config.context_length, self.config.context_length * self.config.patches_per_frame], dim=1)
		elif self.config.tactile and self.config.action:
			action, x_tactile, x = torch.split(x, [self.config.context_length + 1, self.config.context_length, self.config.context_length * self.config.patches_per_frame], dim=1)

		x = x.view(-1, self.config.context_length, int(x.shape[1] / self.config.context_length), x.shape[2])  # input  shape = [bs, t*num_patch, enc_dim] || output shape = [bs, t, num_patch, enc_dim]
		if self.config.action:	x = x.reshape(-1, x.shape[2], x.shape[3])  									  # output shape = [bs*t, num_patch, enc_dim]	
		else:                   x = x.view(-1, x.shape[2], x.shape[3])  									  # ^^
		x = x.transpose(1, 2)  																				  # output shape = [bs*t, enc_dim, num_patch]
		x = x.view(x.shape[0], x.shape[1], patch_size, patch_size)  										  # output shape = [bs*t, enc_dim, patch_size, patch_size]

		x = self.transformer.sig(self.transformer.decode_and_depatch(x)) 									  # input shape  = [bs*t, enc_dim, patch_size, patch_size] || output shape = [bs*t, c, h, w]
		x = x.reshape(-1, self.config.context_length, self.config.input_dim, self.config.H, self.config.W)    # output shape = [bs, t, c, h, w]

		loss = F.l1_loss(x, targets, reduction='mean')

		if self.config.image ==True and self.config.tactile == False:
			total_loss = loss
			tactile_loss = None
			pred_tactile = None

		if self.config.image ==True and self.config.tactile:
			pred_tactile = self.transformer.sig(self.transformer.tactile_debedding(x_tactile))
			tactile_loss = F.l1_loss(pred_tactile, tactile_targets, reduction='mean')
			total_loss = (1.0*loss) + (1.0*tactile_loss)

		return x, pred_tactile, total_loss, loss, tactile_loss

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def get_num_params(self, non_embedding=True):
		n_params = sum(p.numel() for p in self.parameters())
		if non_embedding:
			n_params -= self.transformer.wpe.weight.numel()
		return n_params

	def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
		if self.config.freeze_image_model or self.config.freeze_ac_image_model:
			for param in self.transformer.prior_model.parameters():
				param.requires_grad = False

		if self.config.freeze_image_tokenizer:
			for param in self.transformer.patch_and_embed.parameters():
				param.requires_grad = False

		if self.config.freeze_image_decoder:  # not sure if you can actually freeze the decoder, its not really a thing maybe?
			for param in self.transformer.decode_and_depatch.parameters():
				param.requires_grad = False
	
		param_dict = {pn: p for pn, p in self.named_parameters()}                 # start with all of the candidate parameters
		param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}   # filter out those that do not require grad
		# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no. i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
		optim_groups = [ {'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0} ]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
		print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
		# Create AdamW optimizer and use the fused version if it is available
		fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and device_type == 'cuda:1'  # was just cuda
		extra_args = dict(fused=True) if use_fused else dict()
		optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
		print(f"using fused AdamW: {use_fused}")

		return optimizer
