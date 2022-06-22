"""

Adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

"""
import torch
from torch.nn import ReLU


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, robust_model):
        self.model = model
        self.robust_model = robust_model
        self.gradients = None
        self.robust_gradients = None
        self.forward_relu_outputs = []
        self.forward_relu_outputs_robust = []
        # Put model in evaluation mode
        self.model.eval()
        self.robust_model.eval()
        self.update_relus()
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

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            # print('relu backward hook')
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        def relu_backward_hook_function_robust(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs_robust[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs_robust[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function_robust(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs_robust.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.model.model._modules.items():
            if isinstance(module, ReLU) and pos != 'fc':
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

        for pos, module in self.robust_model.model.model._modules.items():
            if isinstance(module, ReLU) and pos != 'fc':
                module.register_backward_hook(relu_backward_hook_function_robust)
                module.register_forward_hook(relu_forward_hook_function_robust)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        # input = torch.tensor(input_image)
        self.model.zero_grad()
        self.robust_model.zero_grad()
        input_image.requires_grad_()

        with torch.enable_grad():
            model_output = self.model(input_image)
            robust_model_output = self.robust_model(input_image)

            ensemble_output = ((model_output.softmax(1) + robust_model_output.softmax(1))*0.5).log()

        gradient = torch.autograd.grad(ensemble_output[0][target_class], [input_image])[0].detach()
        return gradient.detach().cpu()

