import os
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from torch import nn


# append all the conv layers and their respective weights to the list

transformer = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(
        #mean=[0.485, 0.456, 0.406],
        #std=[0.229, 0.224, 0.225]
    #)
])


def get_feacture_map():
    model = models.resnet50(pretrained=True)
    print(model)
    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    model_children = list(model.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")
    return conv_layers

def main():
    conv_layers=get_feacture_map()
    dir_path='leaflets'
    file_names = os.listdir(dir_path)
    num_layer = 40
    if not os.path.exists('output'):

        os.makedirs('output',mode=0o777)
    out_path = "./output\\"

    for file in file_names:
        #print(os.path.join(dir_path, file))
        image_path = os.path.join(dir_path, file)
        image = Image.open(image_path)
        image = transformer(image).unsqueeze(0)
        results = [conv_layers[0](image)]
        for i in range(1, len(conv_layers)):
            results.append(conv_layers[i](results[-1]))

        outputs = results[num_layer]
        plt.figure(figsize=(5, 5))
        layer_viz = outputs[0, :, :, :]
        layer_viz = layer_viz.data
        filter=layer_viz[38]


        plt.imshow(filter, cmap='gray')
        plt.axis("off")
        print(f"Saving layer {file} feature maps...")
        name = out_path +file + ".png"
        plt.savefig(name)

        plt.close()


if __name__ == '__main__':
    main()