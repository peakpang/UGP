import torch
import torch.nn as nn

from ugp.modules.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample


class KPConvFPN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4,
            init_dim * 4,
            kernel_size,
            init_radius * 2,
            init_sigma * 2,
            group_norm,
            strided=True,
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8,
            init_dim * 8,
            kernel_size,
            init_radius * 4,
            init_sigma * 4,
            group_norm,
            strided=True,
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.encoder5_1 = ResidualBlock(
            init_dim * 16,
            init_dim * 16,
            kernel_size,
            init_radius * 8,
            init_sigma * 8,
            group_norm,
            strided=True,
        )
        self.encoder5_2 = ResidualBlock(
            init_dim * 16, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder5_3 = ResidualBlock(
            init_dim * 32, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )

        self.decoder4 = UnaryBlock(init_dim * 48, init_dim * 16, group_norm)
        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        feats_s5 = self.encoder5_1(feats_s4, points_list[4], points_list[3], subsampling_list[3])
        feats_s5 = self.encoder5_2(feats_s5, points_list[4], points_list[4], neighbors_list[4])
        feats_s5 = self.encoder5_3(feats_s5, points_list[4], points_list[4], neighbors_list[4])

        latent_s5 = feats_s5
        feats_list.append(feats_s5)

        latent_s4 = nearest_upsample(latent_s5, upsampling_list[3])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4(latent_s4)
        feats_list.append(latent_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list

class KPConvFPN_encoder(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN_encoder, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4,
            init_dim * 4,
            kernel_size,
            init_radius * 2,
            init_sigma * 2,
            group_norm,
            strided=True,
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8,
            init_dim * 8,
            kernel_size,
            init_radius * 4,
            init_sigma * 4,
            group_norm,
            strided=True,
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.encoder5_1 = ResidualBlock(
            init_dim * 16,
            init_dim * 16,
            kernel_size,
            init_radius * 8,
            init_sigma * 8,
            group_norm,
            strided=True,
        )
        self.encoder5_2 = ResidualBlock(
            init_dim * 16, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder5_3 = ResidualBlock(
            init_dim * 32, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )


    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        feats_s5 = self.encoder5_1(feats_s4, points_list[4], points_list[3], subsampling_list[3])
        feats_s5 = self.encoder5_2(feats_s5, points_list[4], points_list[4], neighbors_list[4])
        feats_s5 = self.encoder5_3(feats_s5, points_list[4], points_list[4], neighbors_list[4])

    
        feats_list.append(feats_s5)
        feats_list.append(feats_s4)
        feats_list.append(feats_s3)
        feats_list.append(feats_s2)
        feats_list.reverse()
        return feats_list

class KPConvFPN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN_decoder, self).__init__()
        self.inproj = nn.Linear(input_dim,init_dim * 32)
        self.decoder4 = UnaryBlock(init_dim * 48, init_dim * 16, group_norm)
        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        upsampling_list = data_dict['upsampling']

        feats_s2 = feats[0]
        feats_s3 = feats[1]
        feats_s4 = feats[2]
        feats_s5 = feats[3]
        feats_s5 = self.inproj(feats_s5)
        latent_s4 = nearest_upsample(feats_s5, upsampling_list[3])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4(latent_s4)


        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)


        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2) # feats_f的特征

        return latent_s2

class ConvBlockU(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlockU,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)
        self.norm1 = nn.InstanceNorm2d(out_channels,affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)
        self.norm2 = nn.InstanceNorm2d(out_channels,affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        else:
            self.residual_conv = None
        
    def forward(self,x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.norm2(y)
        
        if self.residual_conv is not None:
            x = self.residual_conv(x)
        y = y + x
        y = self.relu2(y)

        return y    
    
class BEV_encoder(torch.nn.Module):
    def __init__(self,input_dim=1,init_dim=64):
        super(BEV_encoder,self).__init__()

        self.encoder1 = ConvBlockU(input_dim,init_dim)
        self.encoder2 = ConvBlockU(init_dim,init_dim*2)
        self.encoder3 = ConvBlockU(init_dim*2,init_dim*4)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
    
    def forward(self,x):
        feats_list=[]
        
        x1 = self.encoder1(x)   
        x1_p = self.maxpool(x1) 
        feats_list.append(x1)
        x2 = self.encoder2(x1_p)
        x2_p = self.maxpool(x2) 
        feats_list.append(x2)
        x3 = self.encoder3(x2_p)       
        feats_list.append(x3) 

        return feats_list