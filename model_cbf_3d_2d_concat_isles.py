import torch
import torch.nn as nn

class ldct_32d_net(nn.Module):
    def __init__(self):
        super(ldct_32d_net, self).__init__()
# (inp channels, Out Channels, Kernel Size, Stride, Padding)
        
        # Down 1
        
        self.conv10 = nn.Sequential(nn.Conv3d(1,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())
        
        self.conv11 = nn.Sequential(nn.Conv3d(32,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())
        
        self.down1  = nn.MaxPool3d(2,2)
        
        # Down 2
        
        self.conv20 = nn.Sequential(nn.Conv3d(32,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())
        
        self.conv21 = nn.Sequential(nn.Conv3d(64,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())
        
        self.down2  = nn.MaxPool3d(2,2)

        # Down 3
        
        self.conv30 = nn.Sequential(nn.Conv3d(64,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())
        
        self.conv31 = nn.Sequential(nn.Conv3d(128,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())
        
        self.down3  = nn.MaxPool3d(2,2)

         # Down 4
        
        self.conv40 = nn.Sequential(nn.Conv3d(128,256,3,1,1),nn.BatchNorm3d(256),
        nn.ReLU())
        
        self.conv41 = nn.Sequential(nn.Conv3d(256,256,3,1,1),nn.BatchNorm3d(256),
        nn.ReLU())
        
        self.down4  = nn.MaxPool3d(2,2)

        # Middle

        self.convm1 = nn.Sequential(nn.Conv2d(256,512,3,1,1),nn.BatchNorm2d(512),
        nn.ReLU())
        
        self.convm2 = nn.Sequential(nn.Conv2d(512,512,3,1,1),nn.BatchNorm2d(512),
        nn.ReLU())

        # Up 1

        self.up1   = nn.ConvTranspose2d(512,256,2,2)
        
        self.conv50 = nn.Sequential(nn.Conv2d(512,256,3,1,1),nn.BatchNorm2d(256),
        nn.ReLU(inplace = True))
        
        self.conv51 = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.BatchNorm2d(256),
        nn.ReLU(inplace = True))

        # Up 2

        self.up2   = nn.ConvTranspose2d(256,128,2,2)

        self.conv60 = nn.Sequential(nn.Conv2d(256,128,3,1,1),nn.BatchNorm2d(128),
        nn.ReLU(inplace = True))
        self.conv61 = nn.Sequential(nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),
        nn.ReLU(inplace = True))

        # Up 3

        self.up3   = nn.ConvTranspose2d(128,64,2,2)

        self.conv70 = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.BatchNorm2d(64),
        nn.ReLU(inplace = True))
        
        self.conv71 = nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),
        nn.ReLU(inplace = True))

        # Up 4

        self.up4   = nn.ConvTranspose2d(64,32,2,2)

        self.conv80 = nn.Sequential(nn.Conv2d(64,32,3,1,1),nn.BatchNorm2d(32),
        nn.ReLU(inplace = True))
        
        self.conv81 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),
        nn.ReLU(inplace = True))
        
      

        self.out = nn.Conv2d(32,1,1,1)
        self.skip = nn.Identity()
        self._init_weights()
        
    def _init_weights(self):        
         for m in self.modules():             
            if isinstance(m, nn.Conv3d):
               torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
               m.bias.data.fill_(0.01)   
            elif isinstance(m, nn.Conv2d):
               torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
               m.bias.data.fill_(0.01)
               
    def forward(self, x):
        
        x10 = self.conv10(x)
       	x11 = self.conv11(x10)
       	x_d1 = self.down1(x11)
        
       	x20 = self.conv20(x_d1)
       	x21 = self.conv21(x20)
       	x_d2 = self.down2(x21)        
        
       	x30 = self.conv30(x_d2)
       	x31 = self.conv31(x30)
       	x_d3 = self.down3(x31)        
               
       	x40 = self.conv40(x_d3)
       	x41 = self.conv41(x40)
       	x_d4 = self.down4(x41)
        
        x_d4 = torch.squeeze(torch.mean(x_d4,2),2)
       	xm1 = self.convm1(x_d4)
        	
        x_u1 = self.up1(self.convm2(xm1))
            	
       	#x_u1 = self.up1(xm1)        
        
        x_c1 = torch.cat((torch.squeeze(torch.mean(x41,2),2) ,x_u1), dim = 1)
        #print(x_c1.size())
       	x50 = self.conv50(x_c1)
       	x51 = self.conv51(x50)
       	x_u2 = self.up2(x51)        
       
        x_c2 = torch.cat((torch.squeeze(torch.mean(x31,2),2) ,x_u2), dim = 1)
       	
        x60 = self.conv60(x_c2)
       	x61 = self.conv61(x60)
       	x_u3 = self.up3(x61)
        
        x_c3 = torch.cat((torch.squeeze(torch.mean(x21,2),2) ,x_u3), dim = 1)
       	x70 = self.conv70(x_c3)
       	x71 = self.conv71(x70)
       
       	x_u4 = self.up4(x71)
           
        x_c4 = torch.cat((torch.squeeze(torch.mean(x11,2),2) ,x_u4), dim = 1)
       	x80 = self.conv80(x_c4)
        x81 = self.conv81(x80)
       
        y = self.out(x81)
       
        
        return y
#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%%
# x = torch.rand(1,1,30,64,64)
# net = ldct_32d_net()

# y = net(x)
# print(y.size())

# p = count_parameters(net)
# print(p)