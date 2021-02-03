close all;
clear;clc
load('test_residue.mat')
load('mean_mask4.mat')
x = squeeze(squeeze(test_residue(1,:,:,:)));
x = imrotate(x,270);
x = double(x);
y = flip(double(x),2);
y = y.*mask;
y = 6000*y;
y(y<0) = 0;
ctshow(y)




