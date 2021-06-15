

Computed Tomography (CT) Perfusion imaging is a non-invasive medical imaging modality that has also established itself as a fast and economical imaging modality for diagnosing cerebrovascular diseases such as acute ischemia, subarachnoid hemorrhage, and vasospasm. Current CT perfusion imaging being dynamic in nature, requires three-dimensional data acquisition at multiple time points (temporal), resulting in a high dose for the patient under investigation. Low-dose CT perfusion (CTP) imaging suffers from low-quality perfusion maps as the noise in CTP data being spectral in nature. The thesis attempts to develop methods that are fully data-driven and deep learning-based to obtain improved perfusion maps directly from low-dose CT Perfusion data.

The inverse problem of obtaining high-quality perfusion maps from low-dose CT Perfusion data is a well-known ill-posed inverse problem. The present state-of-the-art techniques are computationally expensive and necessitate explicit information about the Arterial Input Function (AIF). To combat the same, we propose a novel deep learning-based end-to-end framework to produce high-quality Cerebral Blood Flow (CBF) maps from low-dose raw CTP data. The proposed models can perform the deconvolution without explicit information of the Arterial Input Function (AIF) and are not susceptible to varying levels of noise. Detailed experimentation and their results validated the superiority of the proposed deep learning framework over the existing state-of-the-art algorithms. 

The proposed architecture has a major bottleneck as it can not handle variable number of time points data. In the next part of the thesis, a novel hybrid network combining the benefits of three-dimensional (3D) and two-dimensional (2D) convolutions to handle variable number of time/temporal points was developed. It also performs deconvolution without explicit information on the Arterial Input Function (AIF). This is the first network that can handle variable time points dynamic 2D data and thus it can be extended to analogous modalities like DCE-MRI.

Both these methods are fully data-driven and aimed at working with less training data, thus having a good appeal for clinical settings. The standard approach of obtaining CT perfusion maps from temporal domain to map domain involves a significant number of preprocessing steps. The developed methods are single-step procedures and provide fast processing without compromising the quality of the perfusion maps for low-dose CT perfusion imaging. Integrating these methods with the post-processing software platforms will enable the availability of high-quality perfusion maps especially for time critical operations like ischemic stroke imaging. 


# Deep-Robust-Deconvolution-of-Low-Dose-CT-Perfusion

Data was prepared in accordance to the pre-processing pipeline presented in https://github.com/ruogufang/SPD.

Our proposed algorithm was compared against Online-Sparse Perfusion Deconvolution algorithm presented in - Towards robust deconvolution of low-dose perfusion CT: Sparse perfusion deconvolution using online dictionary learning - 
Ruogu Fang, Tsuhan Chen and Pina C. Sanelli - Medical image analysis, Elsevier 2013 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4196260/

Note - The raw measurement data for the experiments is not provided in this repository.

Requirements for the codes:

1. Python 3.7
2. Pytorch 1.3.1
3. Matlab 2018b

Feel free to drop a mail to arindamdutta1996@gmail.com incase you find any bug in the codes or need any help with understanding them. 
