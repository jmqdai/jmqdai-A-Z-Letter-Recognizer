import torch.nn as nn
import timm
import torch

class LetterClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = timm.create_model('efficientnet_b0', pretrained = False)
        features = list(self.base.children())[:-1]
        self.features = nn.Sequential(*features)
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location = 'cpu')

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # strip 'module.' if present
    new_state = {}
    for k,v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    try:
        model.load_state_dict(new_state, strict = True)
    except Exception as e:
        print("Strict load failed:", e)
        missing, unexpected = model.load_state_dict(new_state, strict = False)
        print("Loaded with strict = False")
        if missing: print("Missing keys:", missing)
        if unexpected: print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model
