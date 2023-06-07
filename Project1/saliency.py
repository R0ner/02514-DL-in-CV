import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def add_noise(image, noise_level):
    ''' 
    Add gaussian noise to image
    '''
    return image + torch.randn_like(image) * noise_level

def compute_vanilla_grad(model, image, label):
    '''
    Compute vanilla gradient saliency map
    source: 
    '''
    output = model(image)
    output[0, label].backward()

    # Smoothgrad authors recommend abs() for imagenet: https://arxiv.org/pdf/1706.03825.pdf
    saliency_map = image.grad.abs().squeeze().detach().cpu().numpy()
    return saliency_map

def compute_smoothgrad(model, image, label, noise_level=0.1, num_samples=30):
    '''
    Compute SmoothGrad gradient saliency map
    source: https://arxiv.org/pdf/1706.03825.pdf
    Notes:
     - Authors recommend noise level 10%-20%
     - Authors detect diminishing return when num_samples > 50%
    '''
    accumulated_gradients = torch.zeros_like(image)
    for _ in range(num_samples):
        noisy_image = add_noise(image.detach(), noise_level)
        noisy_image.requires_grad = True

        output = model(noisy_image)
        output[0, label].backward()

        accumulated_gradients += noisy_image.grad

    smooth_grad = accumulated_gradients / num_samples 
    # Smoothgrad authors recommend abs() for imagenet: https://arxiv.org/pdf/1706.03825.pdf
    saliency_map = smooth_grad.abs().squeeze().detach().cpu().numpy()
    return saliency_map

def visualize_saliency_map(model, dataloader, method='vanilla', noise_level=0.1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''
    Computes and visualize saliency maps of correcly and incorrectly classifications.
    Implementation currently supports three types of saliency maps:
    - Vanilla grad
    - SmoothGrad
    - gradCAM TODO

    Parameters:
    - model (torch.nn.Module): The trained PyTorch model for which to visualize saliency maps.
    - device (torch.device): The device (CPU or GPU) where the model and data are.
    - dataloader (torch.utils.data.DataLoader): The data loader providing the images and labels.
    '''
    model.eval()
    dataiter = iter(dataloader)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # We iterate through test observations until we find a correct and wrong classification
    correct_flag = False
    incorrect_flag = False

    while not (correct_flag and incorrect_flag):
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct_preds = (preds == labels)
        incorrect_preds = (preds != labels)

        if correct_preds.any() and not correct_flag:

            model.zero_grad()
            correct_image = images[correct_preds][0].unsqueeze(0).detach()
            correct_image.requires_grad = True
            correct_label = labels[correct_preds][0]

            if method.lower() == 'vanilla':
                correct_saliency_map = compute_vanilla_grad(model, correct_image, correct_label)
            elif method.lower() == 'smoothgrad':
                correct_saliency_map = compute_smoothgrad(model, correct_image, correct_label, noise_level=noise_level)
            else:
                raise ValueError(f'Unknown method {method}')

            axes[0, 0].imshow(correct_image.detach().cpu().squeeze().numpy(), cmap='gray')
            axes[0, 0].title.set_text('Correct Classification')
            axes[0, 0].axis('off')
            axes[0, 1].imshow(correct_saliency_map, cmap='hot')
            axes[0, 1].title.set_text('Saliency Map')
            axes[0, 1].axis('off')

            correct_flag = True


        if incorrect_preds.any() and not incorrect_flag:
            model.zero_grad()
            incorrect_image = images[incorrect_preds][0].unsqueeze(0).detach()
            incorrect_image.requires_grad = True
            incorrect_label = labels[incorrect_preds][0]

            if method.lower() == 'vanilla':
                incorrect_saliency_map = compute_vanilla_grad(model, incorrect_image, incorrect_label)
            elif method.lower() == 'smoothgrad':
                incorrect_saliency_map = compute_smoothgrad(model, incorrect_image, incorrect_label, noise_level=noise_level)
            else:
                raise ValueError(f'Unknown method {method}')

            axes[1, 0].imshow(incorrect_image.detach().cpu().squeeze().numpy(), cmap='gray')
            axes[1, 0].axis('off')
            axes[1, 0].title.set_text('Incorrect Classification')
            axes[1, 1].imshow(incorrect_saliency_map, cmap='hot')
            axes[1, 1].axis('off')
            axes[1, 1].title.set_text('Saliency Map')

            incorrect_flag = True            
    fig.suptitle(f"Saliency maps ({method})")
    plt.tight_layout()
    plt.show()




