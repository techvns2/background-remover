import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from model.u2net import U2NET  # Make sure model.py is in model/ folder

def normalize_prediction(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def remove_background(input_path, output_path):
    image = Image.open(input_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Load U2NET model
    model_dir = os.path.join('model', 'u2net.pth')
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.eval()

    with torch.no_grad():
        d1, _, _, _, _, _, _ = net(image_tensor)
        pred = d1[:, 0, :, :]
        pred = normalize_prediction(pred)
        mask = pred.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask).resize(image.size, Image.BILINEAR)
        mask = np.array(mask)

        image = image.convert("RGBA")
        data = np.array(image)
        data[:, :, 3] = mask
        output = Image.fromarray(data)
        output.save(output_path)
