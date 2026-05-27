import torch
import torchvision

def save_checkpoint(state, filename):
    print('=> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_boundary_mask(mask, radius=2):
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 2*radius+1, 2*radius+1),
                        device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=radius) > 0
    eroded  = F.conv2d(mask, kernel, padding=radius) == kernel.numel()
    return (dilated ^ eroded).squeeze()
