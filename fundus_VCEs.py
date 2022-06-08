import random

from matplotlib.colors import LinearSegmentedColormap
from torchvision.utils import save_image

from utils.arguments import get_arguments
from utils_svces.functions import blockPrint
import torch
import numpy as np
import os
import pathlib
import matplotlib as mpl
mpl.use('Agg')
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import trange
from time import sleep
from utils_svces.train_types.helpers import create_attack_config, get_adversarial_attack
import cv2 as cv
from utils_svces.get_config import get_config
from utils_svces.Evaluator import Evaluator


hps = get_arguments()

if not hps.verbose:
    blockPrint()

def float_(x):
    try:
       return float(x)
    except:
        return x

def selective_mask_t(image_src, mask):
    mask = mask.permute((2, 0, 1))
    mask = torch.sgn(torch.sum(mask, dim=0)).to(dtype=image_src.dtype).unsqueeze(0)
    # return mask * image_src
    return mask

def get_image_mask_label(image_name):
    label = int(image_name.split('_')[-1].split('.')[0])
    img = transforms.ToTensor()(Image.open(os.path.join('input_images', image_name)))
    mask_file = os.path.join('input_images', image_name.replace('.png', '_mask.png'))
    with Image.open(mask_file) as mask:
        mask = np.asarray(mask)
        mask = cv.resize(mask, (224, 224))
        mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        # print(f'Mask : {mask.shape}')
        mask = selective_mask_t(img, torch.tensor(mask))

    return img, mask, label

if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
    num_devices = 1
elif len(hps.gpu)==1:
    hps.device_ids = [int(hps.gpu[0])]
    device = torch.device('cuda:' + str(hps.gpu[0]))
    num_devices = 1
else:
    hps.device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(hps.device_ids)))
    num_devices = len(hps.device_ids)
hps.device = device


img_size = 224
num_imgs = hps.num_imgs
out_dir = 'FundusVCEs'
dataset = 'imagenet'
mode = 'examples'
bs = hps.batch_size * len(hps.device_ids)

torch.manual_seed(hps.seed)
random.seed(hps.seed)
np.random.seed(hps.seed)

in_labels = ['healthy', 'DR']

accepted_wnids = []

targets = list(range(2))

some_vces = {f'{str(i)}_1.png':targets for i in range(1, 7)}
top_abs_quantile = lambda tensor_, quantile_=0.9: torch.where(tensor_.abs() >= tensor_.abs().quantile(quantile_), tensor_, torch.tensor([0.0]))

def _plot_counterfactuals(dir, original_imgs, orig_labels, segmentations, targets,
                          perturbed_imgs, perturbed_probabilities, original_probabilities, radii, class_labels, filenames=None, img_idcs=None, num_plot_imgs=hps.num_imgs):
    num_imgs = num_plot_imgs
    num_radii = len(radii)
    scale_factor = 4.0
    target_idx = 0


    if img_idcs is None:
        img_idcs = torch.arange(num_imgs, dtype=torch.long)

    pathlib.Path(dir+'/single_images').mkdir(parents=True, exist_ok=True)

    for lin_idx in trange(len(img_idcs), desc=f'Image write'):
        img_idx = img_idcs[lin_idx]
        num_rows = 2
        num_cols = num_radii + 1
        fig, ax = plt.subplots(num_rows, num_cols,
                               figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
        img_label = orig_labels[img_idx]
        title = f'GT: {class_labels[img_label]}'

        img_segmentation = segmentations[img_idx]
        bin_segmentation = torch.sum(img_segmentation, dim=0) > 0.0
        img_segmentation[:, bin_segmentation] = 0.5
        mask_color = torch.zeros_like(img_segmentation)
        mask_color[1, :, :] = 1.0

        # plot original:
        ax[0, 0].axis('off')
        ax[0, 0].set_title(title)
        img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
        ax[0, 0].imshow(img_original, interpolation='lanczos')

        # plot original with mask
        ax[1, 0].axis('off')
        ax[1, 0].set_title('Difference')

        for radius_idx in range(len(radii)):
            img = torch.clamp(perturbed_imgs[img_idx, target_idx, radius_idx].permute(1, 2, 0), min=0.0,
                              max=1.0)
            img_target = targets[img_idx]
            img_probabilities = perturbed_probabilities[img_idx, target_idx, radius_idx]
            in_probabilities = original_probabilities[img_idx, target_idx, radius_idx]

            target_conf = img_probabilities[img_target]
            target_original = in_probabilities[img_target]
            pred_original = in_probabilities.argmax()
            pred_value = in_probabilities.max()

            ax[target_idx, radius_idx + 1].axis('off')
            ax[target_idx, radius_idx + 1].imshow(img, interpolation='lanczos')

            title = f'{class_labels[img_target]}: {target_conf:.2f}, i:{target_original:.2f},\n p:{class_labels[pred_original]},{pred_value:.2f}'
            ax[target_idx, radius_idx + 1].set_title(title)
            ax[target_idx + 1, radius_idx + 1].axis('off')

            diff = (img_original.cpu() - img.cpu()).abs().sum(2)
            min_diff_pixels = diff.min()
            max_diff_pixels = diff.quantile(0.99)

            diff_scaled = (diff - min_diff_pixels) / (max_diff_pixels - min_diff_pixels)
            diff_scaled = top_abs_quantile(diff_scaled, 0.96)
            cm = LinearSegmentedColormap.from_list('', ['white', 'red'])
            colored_image = cm(diff_scaled.numpy())

            ax[target_idx + 1, radius_idx + 1].imshow(colored_image, interpolation='lanczos')
            title=''
            ax[target_idx + 1, radius_idx + 1].set_title(title)
            save_image(perturbed_imgs[img_idx, target_idx, radius_idx].clip(0, 1), os.path.join(dir, 'single_images', f'{img_idx}.png'))

        plt.tight_layout()
        if filenames is not None:
            fig.savefig(os.path.join(dir, f'{filenames[img_idx]}.png'))
            fig.savefig(os.path.join(dir, f'{filenames[img_idx]}.pdf'))
        else:
            fig.savefig(os.path.join(dir, f'{img_idx}.png'))
            fig.savefig(os.path.join(dir, f'{img_idx}.pdf'))

        plt.close(fig)

plot = False
plot_top_imgs = True

imgs = torch.zeros((num_imgs, 3, img_size, img_size))
masks = torch.zeros((num_imgs, 3, img_size, img_size))
segmentations = torch.zeros((num_imgs, 3, img_size, img_size))
targets_tensor = torch.zeros(num_imgs, dtype=torch.long)
labels_tensor = torch.zeros(num_imgs, dtype=torch.long)
filenames = []

image_idx = 0
kernel = np.ones((5, 5), np.uint8)

selected_vces = list(some_vces.items())


if hps.world_size > 1:
    print('Splitting relevant classes')
    print(f'{hps.world_id} out of {hps.world_size}')
    splits = np.array_split(np.arange(len(selected_vces)), hps.world_size)
    print(f'Using clusters {splits[hps.world_id]} out of {len(targets_tensor)}')


for i, (img_idx, target_classes) in enumerate(selected_vces):
    if hps.world_size > 1 and i not in splits[hps.world_id]:
        pass
    else:
        in_image, mask, label = get_image_mask_label(img_idx)
        for i in range(2):
            targets_tensor[image_idx+i] = target_classes[i]
            labels_tensor[image_idx+i] = label
            imgs[image_idx+i] = in_image
            masks[image_idx + i] = mask
        image_idx += 2
        if image_idx >= num_imgs:
            break


imgs = imgs[:image_idx]
segmentations = segmentations[:image_idx]
targets_tensor = targets_tensor[:image_idx]

use_diffusion = False
for method in [hps.method]:
    if method.lower() == 'svces':
        radii = np.array([float(hps.eps_project)])
        attack_type = 'apgd'
        norm = hps.norm
        use_fw = type(float_(norm)) == float

        stepsize = None
        steps = 75
    else:
        raise NotImplementedError()


    attack_config = create_attack_config(eps=radii[0], steps=steps, stepsize=stepsize, norm=norm, momentum=0.9,
                                         pgd=attack_type, use_fw=use_fw)

    num_classes = len(in_labels)

    img_dimensions = imgs.shape[1:]
    num_targets = 1
    num_radii = len(radii)
    num_imgs = len(imgs)

    with torch.no_grad():

        model_bs = bs
        dir = f'{out_dir}/{mode}/{str(norm)}_classifier_{hps.classifier_type}_{hps.second_classifier_type}_wid_{hps.world_id}_{hps.world_size}_{hps.method}_radius_{hps.eps_project}_seed_{hps.seed}/'
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        out_imgs = torch.zeros((num_imgs, num_targets, num_radii) + img_dimensions)
        out_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
        in_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
        model_original_probabilities = torch.zeros((num_imgs, num_classes))

        n_batches = int(np.ceil(num_imgs / model_bs))

        loss = 'ce-targeted-cfts-conf'
        classifier_config = get_config(hps)
        evaluator = Evaluator(hps, classifier_config, {}, None)

        model = evaluator.load_model(
            hps.classifier_type
        )
        model.to(hps.device)
        model.eval()

        if hps.second_classifier_type != -1:
            second_classifier = evaluator.load_model(
                hps.second_classifier_type
            )
            second_classifier.to(hps.device)
            second_classifier.eval()
            att = get_adversarial_attack(attack_config, model, loss, num_classes,
                                         args=hps, Evaluator=Evaluator,
                                         second_classifier=second_classifier, masks=masks)

            print('setting model to an ensemble classifier')
            model = lambda x: 0.5*(att.model(x).softmax(1)+att.second_classifier(x).softmax(1))
        else:
            second_classifier = None
            att = get_adversarial_attack(attack_config, model, loss, num_classes,
                                         args=hps, Evaluator=Evaluator,
                                         second_classifier=second_classifier, masks=masks)

        for batch_idx in trange(n_batches, desc=f'Batches progress'):
            sleep(0.1)
            batch_start_idx = batch_idx * model_bs
            batch_end_idx = min(num_imgs, (batch_idx + 1) * model_bs)

            batch_data = imgs[batch_start_idx:batch_end_idx, :]
            batch_masks = masks[batch_start_idx:batch_end_idx, :]
            batch_targets = targets_tensor[batch_start_idx:batch_end_idx]
            print('batch segmentations before', segmentations.shape)
            batch_segmentations = segmentations[batch_start_idx:batch_end_idx, :]
            print('batch segmentations after', batch_segmentations.shape)
            target_idx = 0

            orig_out = model(batch_data)
            with torch.no_grad():
                if hps.second_classifier_type != -1:
                    orig_confidences = orig_out
                else:
                    orig_confidences = torch.softmax(orig_out, dim=1)

                model_original_probabilities[batch_start_idx:batch_end_idx, :] = orig_confidences.detach().cpu()

            for radius_idx in range(len(radii)):
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                batch_masks = batch_masks.to(device)

                att.eps = radii[radius_idx]
                att.masks = batch_masks

                batch_adv_samples_i = att.perturb(batch_data,batch_targets,
                                                                best_loss=True)[0].detach()
                out_imgs[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                :] = batch_adv_samples_i.cpu().detach()

                batch_model_out_i = model(batch_adv_samples_i)
                batch_model_in_i = model(batch_data)
                if hps.second_classifier_type != -1:
                    batch_probs_i = batch_model_out_i
                    batch_probs_in_i = batch_model_in_i
                else:
                    batch_probs_i = torch.softmax(batch_model_out_i, dim=1)
                    batch_probs_in_i = torch.softmax(batch_model_in_i, dim=1)

                out_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                :] = batch_probs_i.cpu().detach()
                in_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                :] = batch_probs_in_i.cpu().detach()

            if (batch_idx + 1) % hps.plot_freq == 0 or batch_idx == n_batches-1:
                data_dict = {}

                data_dict['gt_imgs'] = imgs[:batch_end_idx]
                data_dict['gt_labels'] = labels_tensor[:batch_end_idx]
                data_dict['segmentations'] = segmentations[:batch_end_idx]
                data_dict['targets'] = targets_tensor[:batch_end_idx]
                data_dict['counterfactuals'] = out_imgs[:batch_end_idx]
                data_dict['out_probabilities'] = out_probabilities[:batch_end_idx]
                data_dict['in_probabilities'] = in_probabilities[:batch_end_idx]
                data_dict['radii'] = radii
                torch.save(data_dict, os.path.join(dir, f'{num_imgs}.pth'))
                _plot_counterfactuals(dir, imgs[:batch_end_idx], labels_tensor, segmentations[:batch_end_idx],
                                      targets_tensor[:batch_end_idx],
                                      out_imgs[:batch_end_idx], out_probabilities[:batch_end_idx], in_probabilities[:batch_end_idx], radii, in_labels, filenames=None, num_plot_imgs=len(imgs[:batch_end_idx]))
