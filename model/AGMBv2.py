import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


BATCH_NORM_DECAY = 1 - 0.9  # pytorch batch norm `momentum = 1 - counterpart` of tensorflow
BATCH_NORM_EPSILON = 1e-5
type = 0

def get_act(activation):
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation == 'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        # Up is a bilinear interpolation operator
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class BNReLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation='relu', nonlinearity=True, init_zero=False):
        super(BNReLU, self).__init__()

        self.norm = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_DECAY, eps=BATCH_NORM_EPSILON)
        if nonlinearity:
            self.act = get_act(activation)
        else:
            self.act = None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out = self.norm(input)
        if self.act is not None:
            out = self.act(out)
        return out


class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda()
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda()
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std=0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input = input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):
    """
    """

    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative', use_pos=True):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        o = self.self_attention(q=q, k=k, v=v)
        return o



class GTBotBlock(nn.Module):


    def __init__(self, in_dimension, curr_h, curr_w, proj_factor=4, activation='relu', pos_enc_type='relative',
                 stride=1, target_dimension=2048):
        super(GTBotBlock, self).__init__()
        if stride != 1 or in_dimension != target_dimension:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dimension, target_dimension, kernel_size=1, stride=stride),
                BNReLU(target_dimension, activation=activation, nonlinearity=True),
            )
        else:
            self.shortcut = None
        self.curr_h = curr_h
        bottleneck_dimension = target_dimension // proj_factor
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=1, stride=1),
            BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
        )

        self.mhsa = MHSA(in_channels=bottleneck_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,
                         pos_enc_type=pos_enc_type)
        conv2_list = []
        if stride != 1:
            assert stride == 2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )

        self.last_act = get_act(activation)


    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = self.conv1(x)
        Q_h = Q_w = self.curr_h
        N, C, H, W = out.shape
        P_h, P_w = H // Q_h, W // Q_w
        
        if type == 1:
            out = out.reshape(N, C, P_h, Q_h, P_w, Q_w)
            out = out.permute(0, 2, 4, 1, 3, 5)
        out = out.reshape(N * P_h * P_w, C, Q_h, Q_w)
        out = self.mhsa(out)
        out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
        
        out = self.conv2(out)
        N1, C1, H1, W1 = out.shape
        if type == 1:
            out = out.reshape(N , P_h , P_w, C, int(Q_h/2), int(Q_w/2))
            out = out.permute(0, 3, 1, 4, 2, 5)
        out = out.reshape(N, C1, int(H1 * (N1 / N) ** 0.5), int(W1 * (N1 / N) ** 0.5))
        out = self.conv3(out)

        out += shortcut
        out = self.last_act(out)

        return out



class GTBotBlock2(nn.Module):

    def __init__(self, in_dimension, curr_h, curr_w, proj_factor=4, activation='relu', pos_enc_type='relative',
                 stride=1, target_dimension=2048):
        super(GTBotBlock2, self).__init__()
        if stride != 1 or in_dimension != target_dimension:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dimension, target_dimension, kernel_size=1, stride=stride),
                BNReLU(target_dimension, activation=activation, nonlinearity=True),
            )
        else:
            self.shortcut = None
        self.curr_h = curr_h 
        bottleneck_dimension = target_dimension // proj_factor
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=1, stride=1),
            BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
        )

        self.mhsa = MHSA(in_channels=bottleneck_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,
                         pos_enc_type=pos_enc_type)
        conv2_list = []
        if stride != 1:
            assert stride == 2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )

        self.last_act = get_act(activation)


    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x


        out = self.conv1(x)

        Q_h = Q_w = self.curr_h
        N, C, H, W = out.shape
        P_h, P_w = H // Q_h, W // Q_w

        if type == 1:
            out = out.reshape(N, C, P_h, Q_h, P_w, Q_w)
            out = out.permute(0, 2, 4, 1, 3, 5)
        out = out.reshape(N * P_h * P_w, C, Q_h, Q_w)
        out = self.mhsa(out)
        out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
        
        
        out = self.conv2(out)
        N1, C1, H1, W1 = out.shape
        if type == 1:
            out = out.reshape(N , P_h , P_w, C, int(Q_h/2), int(Q_w/2))
            out = out.permute(0, 3, 1, 4, 2, 5)
        out = out.reshape(N, C1, int(H1 * (N1 / N) ** 0.5), int(W1 * (N1 / N) ** 0.5))

        out = self.conv3(out)
        out += shortcut
        out = self.last_act(out)

        return out



class BotBlock(nn.Module):

    def __init__(self, in_dimension, curr_h, curr_w, proj_factor=4, activation='relu', pos_enc_type='relative',
                 stride=1, target_dimension=2048):
        super(BotBlock, self).__init__()
        if stride != 1 or in_dimension != target_dimension:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dimension, target_dimension, kernel_size=1, stride=stride),
                BNReLU(target_dimension, activation=activation, nonlinearity=True),
            )
        else:
            self.shortcut = None

        bottleneck_dimension = target_dimension // proj_factor
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=1, stride=1),
            BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
        )

        self.mhsa = MHSA(in_channels=bottleneck_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,
                         pos_enc_type=pos_enc_type)
        conv2_list = []
        if stride != 1:
            assert stride == 2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )

        self.last_act = get_act(activation)

    def forward(self, x):
        out = self.conv1(x)

        out = self.mhsa(out)
        out = out.permute(0, 3, 1, 2)  # back to pytorch dim order

        out = self.conv2(out)
        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = self.last_act(out)
        return out


CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64

class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=True,
                 norm_layer=nn.BatchNorm2d):
        super(SCBottleneck, self).__init__()
        #group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes*4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes*4)
            )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        
        """print("this is x size:")
        print(x.size())"""
        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += self.shortcut(residual)   # can sha mo kuai
        out = self.relu(out)

        return out



class AGGT(nn.Module):

    def __init__(self, block, num_blocks, class_names=3):
        super().__init__()
        self.in_channels = 128
        self.cardinality = 1
        self.bottleneck_width = 32
        # ResNet-D params
        self.avg_down = False
        self.avd = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        #self.downsample = self._make_downsample_layer()
        self.conv5 = self._make_gt_layer2(1024,2048)
        self.conv5_1_x = self._make_layer(block, num_blocks[3], 512, 2)


        planes=1024
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 2))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features= 64 )
        self.sigmoid = nn.Sigmoid()
        self.conv6 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1,padding=1, bias=False),
            BNReLU(2048, activation='relu', nonlinearity=True)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, class_names)

    def forward(self, x):
    
        x = x.type(torch.FloatTensor)
        x = x.cuda()
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        output1 = self.conv5(x)
        output2 = self.conv5_1_x(x)

        output = torch.cat([output1, output2], 1)

        #Branch Fusion Module

        out = self.globalAvgPool(output)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        array=out.cpu().detach().numpy()
        outchanel=torch.zeros(array.shape[0],2048,16,16)

        for i in range(array.shape[0]):
            asum=array
            median=np.median(asum)

            posi=np.where(asum>median)
            if(len(posi[0])!=32):
                n=32-len(posi[0])
                mid=np.where(asum == median)[0][0:n]
                if(len(mid)==1):
                    mid=(mid,)
                posi=(np.append(posi[0],mid),)

            posinew=np.array([])
            for p in range(0,32):
                posinew = np.append(posinew, np.arange(posi[0][p]*64,(posi[0][p]+1)*64))

            outchanel[i] =( output[i, posinew, :, :])

        output=outchanel.cuda()

        output = self.conv6(output)


        x = self.avg(output)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        downsample = None
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            #layers.append(block(self.in_channels, out_channels, stride))
            layers.append(block(self.in_channels, out_channels, stride,downsample = downsample,
                                cardinality = self.cardinality,
                                bottleneck_width = self.bottleneck_width,
                                avd = self.avd, dilation = 1))
            self.in_channels = out_channels * 4  # when the next time, the in_channel is the 4 times of last in_channel

        return nn.Sequential(*layers)

    def _make_last_layer(self, block, out_channels, num_blocks, stride):

        W = H = 32
        dim_in = 1024
        dim_out = 2048

        stage5 = []
        for i in range(3):
            stage5.append(
                BotBlock(in_dimension=dim_in, curr_h=H, curr_w=W, stride=2 if i == 0 else 1, target_dimension=dim_out)
            )
            if i == 0:
                H = H // 2
                W = W // 2
            dim_in = dim_out

        return nn.Sequential(*stage5)

    def _make_gt_layer(self, ch_in, ch_out):

        W = H = 8
        dim_in = ch_in
        dim_out = ch_out

        stage = []
        for i in range(3):
            stage.append(
                GTBotBlock(in_dimension=dim_in, curr_h=H, curr_w=W, stride=2 if i == 0 else 1, target_dimension=dim_out)
            )
            dim_in = dim_out

        return nn.Sequential(*stage)

    def _make_gt_layer2(self, ch_in, ch_out):

        #W = H = 8//2
        W = H = 8
        dim_in = ch_in
        dim_out = ch_out
        


        stage = []
        stage.append(
            GTBotBlock(in_dimension=dim_in, curr_h=4, curr_w=4, stride=2 if 0 == 0 else 1, target_dimension=dim_out)
        )
        dim_in = dim_out
        

        stage.append(
            GTBotBlock2(in_dimension=dim_in, curr_h=H, curr_w=W, stride=2 if 1 == 0 else 1, target_dimension=dim_out)
        )
        dim_in = dim_out
        

        stage.append(
            BotBlock(in_dimension=dim_in, curr_h=2*H, curr_w=2*W, stride=2 if 2 == 0 else 1, target_dimension=dim_out)
        )
        

        return nn.Sequential(*stage)




def AGGT50(out_c):

    return AGGT(SCBottleneck, [3, 4, 6, 3], class_names=out_c)


"""if __name__ == '__main__':
    model = AGGT50(3)
    x = torch.randn(4,3,64,64)
    result = model(x)
    # print(result.size())
    """
    