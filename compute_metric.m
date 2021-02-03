function [RMSE, SSIM, PSNR, SNR] = compute_metric(InImage, GTImage,Mask)

  InImage=InImage/(max(InImage(:)))*100;

  GTImage = GTImage /max(GTImage(:))*100;

    imnoise_diff = abs(InImage-GTImage);

    RMSE = sqrt(mean(reshape(imnoise_diff(Mask).^2,[],1)));

    PSNR = 20 * log10(max(GTImage(Mask))/(RMSE));

    SSIM = ssim(InImage,GTImage,'DynamicRange', 100);
    
    SNR = 10*log10(mean(InImage(Mask(:)).^2)/var(InImage(Mask(:))-GTImage(Mask(:))));

end