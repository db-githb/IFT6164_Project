from common_imports import *

class CNN_min(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block= nn.Sequential(
            nn.Conv2d(1,16,2),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,2),
            nn.MaxPool2d(2),
        )
        self.linear_1= nn.Linear(32*6*6, 200)
        self.dropout = nn.Dropout(0.2)        
        self.linear_2= nn.Linear(200,100)
        self.logit= nn.Linear(100,10)
    
    def forward(self,x):
        x= self.conv_block(x)
        x= x.view(-1, 32*6*6)
        x= F.relu(self.linear_1(x))
        x= self.dropout(x)
        x= F.relu(self.linear_2(x))
        return F.softmax(self.logit(x))
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block= nn.Sequential(
            nn.Conv2d(1,32,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,2),
            nn.MaxPool2d(2),
        )
        self.linear_1= nn.Linear(64*6*6, 200)
        self.linear_2= nn.Linear(200,200)
        self.logit= nn.Linear(200,10)
    
    def forward(self,x):
        x= self.conv_block(x)
        x= x.view(-1, 64*6*6)
        x= F.relu(self.linear_1(x))
        x= F.relu(self.linear_2(x))
        return F.softmax(self.logit(x))
    
## Basic Convolutional structure
# Convolutional layer
class Basic_Conv2d(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=nb_in_channels, out_channels=nb_out_channels, kernel_size=conv_k, stride=conv_stride, padding=conv_pad),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv2d(x)
        return x

# Maxpool Layer
class Basic_Maxpool2d(nn.Module):
    def __init__(self, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.maxpool2d = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_stride, padding=pool_pad),
        )
    
    def forward(self, x):
        x = self.maxpool2d(x)
        return x

# convolutional block with batch normalization 
class Basic_Conv2d_with_batch_norm(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad):
        super().__init__()
        self.conv2d_bn = nn.Sequential(
            nn.Conv2d(in_channels=nb_in_channels, out_channels=nb_out_channels, kernel_size=conv_k, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(nb_out_channels, eps=0.001),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv2d_bn(x)
        return x
    
# Convolutional block
class Conv_Block(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad, pool_k, pool_stride, pool_pad):
        super().__init__()
        self.conv = Basic_Conv2d(nb_in_channels, nb_out_channels, conv_k, conv_stride, conv_pad)
        self.maxpool = Basic_Maxpool2d(pool_k, pool_stride, pool_pad)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x
    

# CNN Model by the passed configuration to constructor (with the configuration of the hidden layers)
class Config_CNN(nn.Module):
    def __init__(self, depth_image, image_size, output_size, nb_conv_block, nb_hidden_layers, conv_k_size,
                 conv_out_channels_start_size, fully_connect_start_size):
        super().__init__()
        self.model_name = 'Configurable CNN' + '_' + str(nb_conv_block) + '_' + str(conv_k_size) + '_' + str(conv_out_channels_start_size)
        self.depth_image = depth_image
        self.image_size = image_size
        self.nb_hidden_layers = nb_hidden_layers
        self.conv_k_size = conv_k_size
        # Convolutional Network
        nb_out_channels = conv_out_channels_start_size
        conv_modules = []
        conv_modules.append(Conv_Block(depth_image,conv_out_channels_start_size,
                                        *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
        for _ in range(nb_conv_block-1):
            conv_modules.append(Conv_Block(nb_out_channels,nb_out_channels*2,
                                            *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
            nb_out_channels *= 2
        conv_modules.append(Basic_Conv2d(nb_out_channels, nb_out_channels*2,
                                            *(conv_k_size,1,int(conv_k_size/2))))
        self.conv_net = nn.Sequential(*conv_modules)
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Fully Connect
        temp_fully_connect_size = fully_connect_start_size
        temp_hidden_list = []
        temp_hidden_list.append(nn.Sequential(
            nn.Linear(self.get_linear_input_size(depth_image, image_size), temp_fully_connect_size),
            nn.ReLU(),
        ))
        for _ in range(nb_hidden_layers-1):
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                nn.ReLU(),
            ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(temp_fully_connect_size, output_size),
        )

    def get_linear_input_size(self, depth_image, image_size):
        rand_input = Variable(torch.rand(1, depth_image, image_size, image_size))
        rand_output = self.conv_net(rand_input)
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size

    
    def forward(self, x):
        x_reshaped = x.view(-1, self.depth_image, self.image_size, self.image_size)
        x_conv = self.conv_net(x_reshaped)
        x_flatten = self.flatten(x_conv)
        x_hidden_layer = self.hidden_layers[0](x_flatten)
        for i in range(1, self.nb_hidden_layers):
            x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
        x_final = self.finalLinear(x_hidden_layer)
        return x_final   

"""
Inception Modules
"""
class Advanced_Stack_block(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, nb_conv, kernel_size):
        # Kernel size must be odd number
        super().__init__()
        reduced_size = int(nb_out_channels/4)
        if reduced_size == 0:
            reduced_size = 1
        self.nb_conv = nb_conv
        self.conv_1x1 = Basic_Conv2d_with_batch_norm(nb_in_channels, nb_out_channels, *(1,1,0))

        self.stack_conv = nn.ModuleList([Basic_Conv2d_with_batch_norm(nb_out_channels, nb_out_channels, *(kernel_size,1,int((kernel_size-1)/2))) for i in range(nb_conv)])
    
    def forward(self, x):
        outputs = []
        output_1x1 = self.conv_1x1(x)
        outputs.append(output_1x1)
        output_temp = output_1x1
        for i in range(self.nb_conv):
            output_temp = self.stack_conv[i](output_temp)
            outputs.append(output_temp)

        return torch.cat(outputs,1) #1 parce que c'est depth concat

class Advanced_Inception_stack_3x3_block(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, nb_conv_3x3):
        super().__init__()
        self.stack_conv_branch = Advanced_Stack_block(nb_in_channels, nb_out_channels, nb_conv_3x3, 3)
        self.reduc_conv_1x1 = Basic_Conv2d_with_batch_norm((nb_conv_3x3+2)*nb_out_channels, nb_in_channels, *(1,1,0))
        
        self.max_pool = Basic_Maxpool2d(*(3,1,1))
        self.max_pool_conv = Basic_Conv2d_with_batch_norm(nb_in_channels, nb_out_channels, *(1,1,0))
        
        self.last_ReLU_activate = nn.ReLU()
    
    def forward(self, x):
        outputs = []
        
        output_stack_branch = self.stack_conv_branch(x)
        outputs.append(output_stack_branch)

        output_pool = self.max_pool(x)
        output_pool = self.max_pool_conv(output_pool)
        outputs.append(output_pool)

        reduc_input = torch.cat(outputs,1)
        reduc_output = self.reduc_conv_1x1(reduc_input)

        final_output = self.last_ReLU_activate(x + reduc_output)

        return final_output

class Advanced_Inception_stack_3x3(nn.Module):
    def __init__(self, depth_image, image_size, output_size, nb_incep_block, nb_conv_3x3, nb_hidden_layers,
                 fully_connect_start_size):
        super().__init__()
        self.model_name = 'Inception_' + str(nb_incep_block) + '_' + str(nb_conv_3x3)
        self.depth_image = depth_image
        self.image_size = image_size
        self.nb_hidden_layers = nb_hidden_layers
        conv_modules = []
        nb_output_channels = 96
        conv_modules.append(Basic_Conv2d_with_batch_norm(depth_image, nb_output_channels, *(1,1,0)))
        for _ in range(nb_incep_block):
            conv_modules.append(Advanced_Inception_stack_3x3_block(nb_output_channels,nb_output_channels,nb_conv_3x3))

        self.conv_net = nn.Sequential(*conv_modules)
        self.flatten = nn.Flatten()
        # Fully Connect
        temp_fully_connect_size = fully_connect_start_size
        temp_hidden_list = []
        temp_hidden_list.append(nn.Sequential(
            nn.Linear(self.get_linear_input_size(depth_image, image_size), temp_fully_connect_size),
            nn.ReLU(),
        ))
        for _ in range(nb_hidden_layers-1):
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                nn.ReLU(),
            ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(temp_fully_connect_size, output_size),
        )

    def get_linear_input_size(self, depth_image, image_size):
        rand_input = Variable(torch.rand(1, depth_image, image_size, image_size))
        rand_output = self.conv_net(rand_input)
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size

    
    def forward(self, x):
        x_reshaped = x.view(-1, self.depth_image, self.image_size, self.image_size)
        x_conv = self.conv_net(x_reshaped)
        x_flatten = self.flatten(x_conv)
        x_hidden_layer = self.hidden_layers[0](x_flatten)
        for i in range(1, self.nb_hidden_layers):
            x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
        x_final = self.finalLinear(x_hidden_layer)
        return x_final 
    


