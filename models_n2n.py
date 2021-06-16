import torch
import torch.nn as nn

    
    
class N2N_UNet(nn.Module):

    def __init__(self):
        super(N2N_UNet, self).__init__()
# (inp channels, Out Channels, Kernel Size, Stride, Padding)
        
        # Down 1
        
        self.conv10 = nn.Sequential(nn.Conv3d(1,48,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1))
        
        self.conv11 = nn.Sequential(nn.Conv3d(48,48,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1))
        
        self.down1  = nn.MaxPool3d(2,2)
        
        # Down 2
        
        self.conv20 = nn.Sequential(nn.Conv3d(48,48,3,1,1),
        nn.LeakyReLU(negative_slope=0.1))
        
        
        self.down2  = nn.MaxPool3d(2,2)

        # Down 3
        
        self.conv30 = nn.Sequential(nn.Conv3d(48,48,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1))
        
        
        self.down3  = nn.MaxPool3d(2,2)

         # Down 4
        
        self.conv40 = nn.Sequential(nn.Conv3d(48,48,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1))
        
        
        self.down4  = nn.MaxPool3d(2,2)


        # Up 1

        self.up1   = nn.ConvTranspose2d(48,48,2,2)
        
        self.conv50 = nn.Sequential(nn.Conv2d(96,96,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))
        
        self.conv51 = nn.Sequential(nn.Conv2d(96,96,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))

        # Up 2

        self.up2   = nn.ConvTranspose2d(96,96,2,2)

        self.conv60 = nn.Sequential(nn.Conv2d(144,96,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))
        
        self.conv61 = nn.Sequential(nn.Conv2d(96,96,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))

        # Up 3

        self.up3   = nn.ConvTranspose2d(96,96,2,2)

        self.conv70 = nn.Sequential(nn.Conv2d(144,96,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))
        
        self.conv71 = nn.Sequential(nn.Conv2d(96,96,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))

        # Up 4

        self.up4   = nn.ConvTranspose2d(96,48,2,2)

        self.conv80 = nn.Sequential(nn.Conv2d(96,64,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))
        
        self.conv81 = nn.Sequential(nn.Conv2d(64,32,3,1,1),
        nn.LeakyReLU(negative_slope = 0.1,inplace = True))
        
      

        self.out = nn.Conv2d(32,1,1,1)
        self.skip = nn.Identity()
        self._init_weights()
        
    
        
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        
        
               
    def forward(self, x):
        
        x10 = self.conv10(x)
       	x11 = self.conv11(x10)
       	x_d1 = self.down1(x11)
        #print(x_d1.size())
       	x20 = self.conv20(x_d1)
       	x_d2 = self.down2(x20)        
        #print(x_d2.size())
       	x30 = self.conv30(x_d2)

       	x_d3 = self.down3(x30)        
        #print(x_d3.size())      
       	x40 = self.conv40(x_d3)
           
       	x_d4 = self.down4(x40)
        #print(x_d4.size())
        x_d4 = torch.squeeze(torch.mean(x_d4,2),2)
        
       	x_u1 = self.up1(x_d4)
        
        #print(x_u1.size())
        x_c1 = torch.cat((torch.squeeze(torch.mean(x40,2),2) ,x_u1), dim = 1)
        
       	x50 = self.conv50(x_c1)
       	x51 = self.conv51(x50)
       	x_u2 = self.up2(x51)        
        #print(x_u2.size())
        x_c2 = torch.cat((torch.squeeze(torch.mean(x30,2),2) ,x_u2), dim = 1)
       	
        x60 = self.conv60(x_c2)
       	x61 = self.conv61(x60)
       	x_u3 = self.up3(x61)
        #print(x_u3.size())
        x_c3 = torch.cat((torch.squeeze(torch.mean(x20,2),2) ,x_u3), dim = 1)
        #print(x_c3.size())
       	x70 = self.conv70(x_c3)
       	x71 = self.conv71(x70)
       
       	x_u4 = self.up4(x71)
        #print(x_u4.size())  
        x_c4 = torch.cat((torch.squeeze(torch.mean(x11,2),2) ,x_u4), dim = 1)
        #print(x_c4.size())
       	x80 = self.conv80(x_c4)
        x81 = self.conv81(x80)
       
        y = self.out(x81)
       
        
        return y
#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%%
#x = torch.rand(1,1,40,64,64)
#net = N2N_UNet()
#
#y = net(x)
#print(y.size())
#
#p = count_parameters(net)
#print(p)
