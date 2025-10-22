import abc
import torch

LOW_RESOURCE = False

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out


        def forward(x, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # print(x.shape) # 2, 4096, 320
            # exit(0)
            batch_size, sequence_length, dim = x.shape
            context = encoder_hidden_states
            mask = attention_mask

            h = self.heads
            # print(h) # 8
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)
            # q = self.reshape_heads_to_batch_dim(q)
            # k = self.reshape_heads_to_batch_dim(k)
            # v = self.reshape_heads_to_batch_dim(v)
            
            
            
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            
            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            # try:
            attn = controller(attn, is_cross, place_in_unet) # here the error
            # except Exception as e:
            #     pass
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            # out = self.reshape_batch_dim_to_heads(out)
            out = self.batch_to_head_dim(out)
            # print(out.shape) # 2, 4096, 320
            # exit(0)
            return to_out(out)

        return forward

        # def forward(x, context=None, mask=None):
        #     batch_size, sequence_length, dim = x.shape
        #     h = self.heads
        #     q = self.to_q(x)
        #     is_cross = context is not None
        #     context = context if is_cross else x
        #     k = self.to_k(context)
        #     v = self.to_v(context)
        #     q = self.reshape_heads_to_batch_dim(q)
        #     k = self.reshape_heads_to_batch_dim(k)
        #     v = self.reshape_heads_to_batch_dim(v)

        #     sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        #     if mask is not None:
        #         mask = mask.reshape(batch_size, -1)
        #         max_neg_value = -torch.finfo(sim.dtype).max
        #         mask = mask[:, None, :].repeat(h, 1, 1)
        #         sim.masked_fill_(~mask, max_neg_value)

        #     # attention, what we cannot get enough of
        #     attn = sim.softmax(dim=-1)
        #     attn = controller(attn, is_cross, place_in_unet)
        #     out = torch.einsum("b i j, b j d -> b i d", attn, v)
        #     out = self.reshape_batch_dim_to_heads(out)
        #     return to_out(out)

        # return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        # print(net_.__class__.__name__)
        # exit(0)
        # if net_.__class__.__name__ == 'CrossAttention':
        # print("net_.__class__.__name__", net_.__class__.__name__)
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            # print("not at all?") # not at all..
            # print("net_.__class__.__name__", net_.__class__.__name__)
            # exit(0)
            return count + 1
        elif hasattr(net_, 'children'):
            
            for net__ in net_.children():
                
                count = register_recr(net__, count, place_in_unet)
        # exit(0)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    # sub_nets = list(model.named_children())
    # sub_nets = model.named_modules()

    # for name, module in model.named_modules():


    for net in sub_nets:
        # print(net[0])
        # print(net[1])
        # exit(0)
        if "down" in net[0]:
            # print("here1")
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            # print("here2")
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            # print("here3")
            cross_att_count += register_recr(net[1], 0, "mid")
    # exit(0)
    controller.num_att_layers = cross_att_count
    print(f"Registered {cross_att_count} attention layers.") # 0
    # exit(0)
