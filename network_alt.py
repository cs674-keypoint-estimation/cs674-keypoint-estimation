import torch

import numpy as np

import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import network
import hydra

# Parameters based on original: class Unsupervised_kpnet(nn.Module) with additional concatenated layer
# future version will have point transformer instead of pointnet
# 1.  Takes as input 1024 features per point
# 2.  Convolution layer outputs 256 gets concatenated with output of original architecture
# 3.  Additionally, dropout is applied to both paths to encourage learning


class Unsupervised_PointNetToKeypointsAlt(torch.nn.Module):
    def __init__(self, cfg):
        super(Unsupervised_PointNetToKeypointsAlt, self).__init__()     
        self.conv_to_concat = torch.nn.Conv1d(1024, 128, 1)
        self.resblock1 = network.residual_block(1024, 512)
        self.resblock2 = network.residual_block(512,256)
        self.resblock3 = network.residual_block(256,128)        
        self.final_conv = torch.nn.Conv1d(256, cfg.key_points, 1) 
        self.softmax = torch.nn.Softmax(dim=2)
        self.pointnet = network.PointNetfeat()
    
    #forward logic
    def forward(self, input):
        #input is [26, 2048, 3])
        # batch size 26, 2048 points per model, 3 dimensions per point
        # to get 2048 on last dim we use permute to effectively transpose last 2 dims
        #https://pytorch.org/docs/stable/generated/torch.permute.html
        input_transposed = torch.permute(input, [0,2,1])        
        x = input_transposed
        x = self.pointnet(x)
        #x shape is [26, 1024, 2048]
   
        # we apply dropout to promote learning in both paths
        # we split a conv layer to concatenate at the end

        x1 = self.conv_to_concat(x)
        #x1 shape is [26, 256, 2048]

        x = self.resblock1(x)
        # x.shape [26, 512, 2048]


        x = self.resblock2(x)
        # x.shape [26, 256, 2048]        
        x = self.resblock3(x)        
        #we concat along our second dimension (dim 1 since we start at 0)
        x = torch.cat([x, x1], dim =1, out=None)
   
        # x.shape [26, 512, 2048]      
        x= self.final_conv(x)     
        # x.shape [26, 10, 2048]               
        probabilities = self.softmax(x)  

        # probabilities.shape [26, 10, 2048]
        output_old = output = torch.matmul(probabilities, input)
        max_prob = torch.argmax(probabilities, 2)
       
        #output.shape [26, 10, 3]
        return (output)
    
class Unsupervised_PointNetToKeypointsWithDropout(torch.nn.Module):
    def __init__(self, cfg):
        super(Unsupervised_PointNetToKeypointsWithDropout, self).__init__()

        self.dropout_p = 0.05
        self.dropout = torch.nn.Dropout(p=self.dropout_p)        
        self.conv_to_concat = torch.nn.Conv1d(1024, 256, 1)
        self.resblock1 = network.residual_block(1024, 512)
        self.resblock2 = network.residual_block(512,256)
        self.final_conv = torch.nn.Conv1d(512, cfg.key_points, 1) 
        self.softmax = torch.nn.Softmax(dim=2)
        self.pointnet = network.PointNetfeat()
    
    #forward logic
    def forward(self, input):
        #input is [26, 2048, 3])
        # to get 2048 on last dim we use permute to effectively transpose last 2 dims
        #https://pytorch.org/docs/stable/generated/torch.permute.html
        input_transposed = torch.permute(input, [0,2,1])        
        x = input_transposed
        x = self.pointnet(x)
        #x shape is [26, 1024, 2048]
   
        # we apply dropout to promote learning in both paths
        # we split a conv layer to concatenate at the end

        x1 = self.conv_to_concat(x)
        #x1 shape is [26, 256, 2048]

        x = self.resblock1(x)
        # x.shape [26, 512, 2048]


        x = self.resblock2(x)
        # x.shape [26, 256, 2048]        
        
        #we concat along our second dimension (dim 1 since we start at 0)
        x = torch.cat([x, self.dropout(x1)], dim =1, out=None)
   
        # x.shape [26, 512, 2048]      
        x= self.final_conv(x)     
        # x.shape [26, 10, 2048]               
        probabilities = self.softmax(x)  
        # probabilities.shape [26, 10, 2048]  
        output = torch.matmul(probabilities, input)

        return (output)


#need this snippet from original network.py code to make cfg work

@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    cfg.split = 'train'
    pc = torch.randn(5, 2048, 3)
    data = [pc, pc, pc, pc]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.sc3k(cfg).to(device) # cuda()   # unsupervised network
    # pdb.set_trace()
    kp1, kp2 = model(data)
    print(kp1.shape, kp1.shape)


if __name__ == '__main__':
    main()