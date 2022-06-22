import argparse
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms
import os

from utils.arguments import get_arguments
from utils_svces.Evaluator import Evaluator
from utils_svces.get_config import get_config
from utils_svces.functions import blockPrint

from cnn_visualizations.misc_functions import (preprocess_image,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_example_params,
                            create_saliency_map)
from cnn_visualizations.integrated_gradients import IntegratedGradients
from cnn_visualizations.guided_backprop_ensemble import GuidedBackprop
# from utils.model_normalization import FundusKaggleWrapper_clahe_v2_new_qual_eval
# from utils.models.models_224x224.resnet_224 import resnet50 as imagenet_resnet50

if __name__ == '__main__':
    hps = get_arguments()
    
    if not hps.verbose:
        blockPrint()
        
    hps.device = torch.device('cpu')

    i=0
    data_folder='input_images'
    result_folder='FundusSaliencyMaps'
    try:
        os.makedirs(result_folder, exist_ok = True)
    except OSError as error:
        print("Directory '%s' can not be created")
     
    for i in np.arange(6):
        (original_image, prep_img, mask_img, target_class, file_name_to_export) =\
            get_example_params(data_folder, i)

    #     device_cpu = torch.device('cpu')
        models = []


        classifier_config = get_config(hps)
        evaluator = Evaluator(hps, classifier_config, {}, None)

        model = evaluator.load_model(hps.classifier_type)
        model.to(hps.device)
        model.eval()

        print(model)

        if hps.second_classifier_type != -1:
            second_classifier = evaluator.load_model(hps.second_classifier_type)
            second_classifier.to(hps.device)
            second_classifier.eval()

        image_size = (224,224)
        test_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        tensors = test_transform(original_image)
        tensors = tensors.unsqueeze(0)

        vis_type_to_cls = {'integrated_gradients': IntegratedGradients, 'guided_backprop': GuidedBackprop}
        for vis_type in vis_type_to_cls.keys():
            filename = f'{result_folder}/{vis_type}_{file_name_to_export}'
            if hps.second_classifier_type != -1: 
                saliency_map_generator = vis_type_to_cls[vis_type](model.cpu(), model.cpu())
                saliency_map = create_saliency_map(saliency_map_generator, tensors, target_class)
                save_gradient_images(original_image, saliency_map, mask_img, filename)
            else:
                saliency_map_generator = vis_type_to_cls[vis_type](model.cpu(), second_classifier.cpu())
                saliency_map = create_saliency_map(saliency_map_generator, tensors, target_class)
                save_gradient_images(original_image, saliency_map, mask_img, filename)



