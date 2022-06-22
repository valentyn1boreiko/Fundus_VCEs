"""

Adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

"""
import torch
import numpy as np


class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model, robust_model):
        self.model = model
        self.robust_model = robust_model
        self.gradients = None
        self.robust_gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.robust_model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        def robust_hook_function(module, grad_in, grad_out):
            self.robust_gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.model.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

        first_layer_robust = list(self.robust_model.model.model._modules.items())[0][1]
        first_layer_robust.register_backward_hook(robust_hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # Forward
        self.model.zero_grad()
        self.robust_model.zero_grad()
        input_image.requires_grad_()
        with torch.enable_grad():
            model_output = self.model(input_image)
            robust_model_output = self.robust_model(input_image)
            ensemble_output = ((model_output.softmax(1) + robust_model_output.softmax(1)) * 0.5).log()
        # Zero grads

        # Target for backprop
        gradient = torch.autograd.grad(ensemble_output[0][target_class], [input_image])[0].detach()
        return gradient.detach().cpu()

    def generate_integrated_gradients(self, input_image, target_class, steps):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        # [0] to get rid of the first channel (1,3,224,224)
        return integrated_grads[0]


