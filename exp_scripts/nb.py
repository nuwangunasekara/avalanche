from skmultiflow.bayes import NaiveBayes

import numpy as np

from torch import nn

import torchvision.models.quantization as models
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms


def create_static_feature_extractor():
    # https://keras.io/api/applications/
    # https://pytorch.org/hub/
    # https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html
    original_model = models.resnet18(pretrained=True, progress=True, quantize=True)
    # you dont need this as the model is quantized
    for param in original_model.parameters():
        param.requires_grad = False

    # Step 1. Isolate the feature extractor.
    model_fe = nn.Sequential(
        original_model.quant,  # Quantize the input
        original_model.conv1,
        original_model.bn1,
        original_model.relu,
        original_model.maxpool,
        original_model.layer1,
        original_model.layer2,
        original_model.layer3,
        original_model.layer4,
        original_model.avgpool,
        original_model.dequant,  # Dequantize the output
    )

    # # Step 2. Create a new "head"
    # new_head = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Linear(num_ftrs, 2),
    # )
    #
    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        model_fe,
        nn.Flatten(1),
        # new_head,
    )
    return new_model


feature_extractor = create_static_feature_extractor()

print(str(feature_extractor))

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

preprocessor = transforms.Compose([
    # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/9
    # expand channels transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if im.mode!='RGB'  else NoneTransform()
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.Pad(0, fill=3),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

X = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
Y = np.array([1, 1, 1, 2, 2, 2])

NB = None

# for x, y in stream.iter_array(X, Y):
for x, y in training_data:
    x = preprocessor(x).unsqueeze(0)

    f = feature_extractor(x)

    if NB is None:
        # nominal_attributes = [i for i in range(f.shape[1])]
        nominal_attributes = None
        NB = NaiveBayes(nominal_attributes=nominal_attributes)

    NB.partial_fit(f, [y])

    p = NB.predict_proba(f)

    # print(y, np.argmax(p_1), p_1)
    print(y, np.argmax(p, axis=1).item())


