import torch.hub
from test import TestTimeFullNet
from argparse import Namespace


ARTICULATED_WEIGHTS = "https://drive.google.com/uc?export=download&id=1bomD88-6N1iGsTtftfGvAm9JeOw8gKwb"


def model_articulated(pretrained=True):
    model_args = Namespace()
    # You can change this parameter for better results (larger for more parts and smaller for fewer parts)
    model_args.alpha = 0.05
    model = TestTimeFullNet(model_args)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(ARTICULATED_WEIGHTS)['model_state'])
    model.cuda().eval()
    return model
