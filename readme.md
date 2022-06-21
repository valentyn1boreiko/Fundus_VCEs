# **Fundus Visual Counterfactual Explanations**

Welcome to the codebase for our MICCAI submission *Visual explanations for the detection of diabetic retinopathy from retinal fundus images.* We will show you how to generate **VCEs** together with their respective **T-VCMs** @ threshold 0.96 on the selected fundus images used in the paper with the ensemble of robust and plain models. 

## Setup

Before we can start with the generation, we have to setup the project and install required packages.

* Start by extracting the content of the .zip file that also contains this readme.md somewhere on your computer. We will refer to the extraction directory as **project_path**.
* Navigate into the  **project_path**

* Download and unzip the weights for robust model trained with TRADES in the l2-ball of radius 0.25 and plain model from [here](https://www.dropbox.com/s/b6oyf4yzfohml0c/FundusModels.zip) into your **project_path**

* Create a new conda env via `conda env create -f python_38_svces_lite.yml`
* Activate the conda environment via `conda activate python_38_dvces`
* Install additionally robustbench via `pip install git+https://github.com/RobustBench/robustbench.git`

## Creating  DVCEs/SVCEs/blended diffusion based VCEs

In the following, we show, how to first set the parameters, and then - generate VCEs of the respective type for 6 selected targets. To choose your own image ids and targets, change `some_vces`.

For any of the proposed parameter settings, feel free to adjust the values, but these are the ones we have used mostly in the paper.

* Generating VCEs for an ensemble of adversarially robust and plain models 
  `seeds=(1);classifier_t=27;second_classifier_type=29;bs=12;nim=12;gpu=1;eps_project=6;norm='L2'`
  and then execute the **starting command**
  `for seed in "${seeds[@]}"; do python fundus_VCEs.py --seed $seed --classifier_type $classifier_t --second_classifier_type $second_classifier_type --gpu $gpu --batch_size $bs --num_imgs $nim --eps_project $eps_project --norm $norm > logs/log; done;` 
* Generating VCEs for an adversarially robust model only 
  `seeds=(1);classifier_t=27;second_classifier_type=-1;bs=12;nim=12;gpu=1;eps_project=6;norm='L2'`
  and then again execute the **starting command**
* Generating VCEs for a plain model only 
  `seeds=(1);classifier_t=29;second_classifier_type=-1;bs=12;nim=12;gpu=1;eps_project=6;norm='L2'`
  and then again execute the **starting command** 

* Important arguments:
    - The batchsize argument `--bs` is the number of samples per gpu, so if you encounter out-of-memory errors you can reduce it without altering results.
    - The number of images argument `--num_imgs` should be greater or equal to the batchsize
    - The norm of the VCEs `--norm`
    - The radius of the VCEs `--eps_project`
    
The resulting images can be found in `FundusVCEs/examples/`.
