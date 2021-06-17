
## Deep-Robust-Deconvolution-of-Low-Dose-CT-Perfusion

Abstract - 

Computed Tomography (CT) Perfusion imaging is a non-invasive medical imaging modality that has also established itself as a fast and economical imaging modality for diagnosing cerebrovascular diseases such as acute ischemia, subarachnoid hemorrhage, and vasospasm. Current CT perfusion imaging being dynamic in nature, requires three-dimensional data acquisition at multiple time points (temporal), resulting in a high dose for the patient under investigation. Low-dose CT perfusion (CTP) imaging suffers from low-quality perfusion maps as the noise in CTP data being spectral in nature. The thesis attempts to develop novel Deep Learning architectures to obtain improved perfusion maps directly from low-dose CT Perfusion data.

![github-small](https://github.com/arindam29/Deep-Robust-Deconvolution-of-Low-Dose-CT-Perfusion/blob/main/exist_tech.png?raw=true)

# Data and Experiments

Data was prepared in accordance to the PCT pre-processing pipeline made available on https://github.com/ruogufang/pct.

Our proposed algorithm was compared against Online-Sparse Perfusion Deconvolution algorithm presented in - Towards robust deconvolution of low-dose perfusion CT: Sparse perfusion deconvolution using online dictionary learning - 
Ruogu Fang, Tsuhan Chen and Pina C. Sanelli - Medical image analysis, Elsevier 2013 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4196260/

Note - The raw measurement data used for the experiments is not provided in this repository.

# Requirements for the codes:

1. Python 3.7
2. Pytorch 1.3.1
3. Matlab 2018b

# Contact
Much like Gradient Descent, I have always believed in learning from mistakes! So, incase you find any bug in the codes or need any help with understanding them - 
Feel free to drop me a mail @ arindamdutta1996@gmail.com.
