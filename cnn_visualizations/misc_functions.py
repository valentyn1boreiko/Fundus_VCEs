"""

Adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

"""
import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


import torch
from torch.autograd import Variable
from torchvision import models
from torch.utils.data import Dataset

def create_saliency_map(saliency_map_generator, tensors, target_class):
    saliency_map = saliency_map_generator.generate_gradients(tensors, target_class)
    saliency_map = saliency_map.squeeze(0).numpy()
    saliency_map = convert_to_grayscale(saliency_map)
    return saliency_map

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    print(im_as_arr.shape)
    grayscale_im = np.sum(abs(im_as_arr) , axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = ((grayscale_im - im_min) / (im_max - im_min)).clip(0,1)
    grayscale_im = (grayscale_im - grayscale_im.min()) / (grayscale_im.max() - grayscale_im.min())
    values = grayscale_im.flatten()
    threshold = np.percentile(values, 96)
    grayscale_im[grayscale_im < threshold] = 0
    return grayscale_im

def normalize(gradient):
    grad_max = gradient.max()
    grad_min = gradient.min()

    grad_min = -max(abs(grad_min), grad_max)
    grad_max = -grad_min

    gradient_normalized = ((gradient - grad_min) / (grad_max - grad_min)).clip(0, 1)

    # gradient_normalized = (gradient_normalized - gradient_normalized.min()) / (gradient_normalized.max()- gradient_normalized.min())
    print(gradient_normalized.max())
    print(gradient_normalized.min())
    return gradient_normalized

def save_gradient_images(img, gradient, mask, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """

    if len(gradient.shape) == 2:
        gradient = gradient * mask
        # gradient[mask == 0] = 0.5
        fig, (ax1, ax2) = plt.subplots(1,2)
        
        ax1.imshow(img)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        cmap = LinearSegmentedColormap.from_list('', ['white', 'red'])
        ax = sns.heatmap(gradient, cmap=cmap, cbar=False, square=True, ax=ax2)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(file_name, bbox_inches='tight')
    else:
        save_image(gradient, file_name)


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=False):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    mean = [0.41859379410743713, 0.28517860174179077, 0.2082202434539795]
    std = [0.30145615339279175, 0.21559025347232819, 0.16515834629535675]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
    #     # im_as_arr[channel] -= mean[channel]
    #     # im_as_arr[channel] /= std[channel]
    # # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    # im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_ten


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    # reverse_mean = [-0.485, -0.456, -0.406]
    # reverse_std = [1/0.229, 1/0.224, 1/0.225]

    reverse_mean = [-0.41859379410743713, -0.28517860174179077, -0.2082202434539795]
    reverse_std = [1/0.30145615339279175, 1/0.21559025347232819, 1/0.16515834629535675]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def get_relevance_scores(saliency_map, annotation):
    print('Relevance scores')
    relevance_within = np.sum(saliency_map[annotation == 1])
    print(relevance_within)
    relevance_total = np.sum(saliency_map)
    print(relevance_total)

    rma = relevance_within / relevance_total
    return rma

def get_relevance_ranks(saliency_map, annotation):
    saliency_map = saliency_map.flatten()
    annotation = annotation.flatten()
    k = int(annotation.sum())

    topK_idx = (-saliency_map).argsort()[:k]
    intersection = annotation[topK_idx].sum()
    relevance_rank_score = intersection / annotation.sum()
    print(relevance_rank_score)
    return relevance_rank_score

def get_example_params(data_folder, example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    mask_folder = data_folder
    classes = [1,1,1,1,1,1]
    example_list = [(f'{idx}_1.png', cls) for idx, cls in zip(np.arange(1,7), classes)]
    mask_list = [f'{idx}_1_mask.png' for idx in np.arange(1,7)]
    img_path = os.path.join(data_folder, example_list[example_index][0])
    print(img_path)
    mask_path = os.path.join(data_folder, mask_list[example_index])
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    file_name_to_export = file_name_to_export + '.png'
    # Read image
    original_image = Image.open(img_path)
    mask = Image.open(mask_path)
    mask = mask.resize((224,224))
    mask = np.asarray(mask)
    mask = mask/255.
    # Process image
    prep_img = preprocess_image(original_image)
    return (original_image,
            prep_img,
            mask,
            target_class,
            file_name_to_export)
