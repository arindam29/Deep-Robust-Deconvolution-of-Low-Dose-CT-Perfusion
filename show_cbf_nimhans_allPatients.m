clear;
clc;
close all;


%%
mA = 50;
z = 7;

%%
load('test_residue.mat')


x = squeeze((squeeze(test_residue(1,:,:,:))));

cd '/media/cds/storage/DATA-1/ARIDAM/Data_CTP/Nimhans'

str = strcat(('p3_z'),num2str(z),'_mA',num2str(mA),'.mat');
load(str);
bb = pct_minBoundingBox(mask_orig);
cd '/media/cds/storage/DATA-1/ARIDAM'



y = double(x);
y = y.*mask_orig;
y = y(bb(1,1):bb(1,2),bb(2,1):bb(2,3),:);
y(y<0) = 0;
y = y/ max(y(:));
y = y*100;

im = (CBF_HD/max(CBF_HD(:)))*100;
imnoise = (CBF_LD/max(CBF_LD(:)))*100;
im_spd = (CBF_ospd/max(CBF_ospd(:)))*100;



im = im(bb(1,1):bb(1,2),bb(2,1):bb(2,3),:);
imnoise = imnoise(bb(1,1):bb(1,2),bb(2,1):bb(2,3),:);


[rmse1,ssim1,psnr1] = compute_metric(imnoise,im,Mask)

[rmse2,ssim2,psnr2] = compute_metric(im_spd,im,Mask)

[rmse3,ssim3,psnr3] = compute_metric(y,im,Mask)

 c = 20;
% 
subplot(1,4,1); ctshow(im,Mask,[0 c]);
subplot(1,4,2); ctshow(imnoise,Mask,[0 c]);
subplot(1,4,3); ctshow(im_spd,Mask,[0 c]);
subplot(1,4,4); ctshow(y,Mask,[0 c]);