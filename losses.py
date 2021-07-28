import torch
import torchvision


class VGGLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGLoss, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:18].eval())
        blocks.append(vgg.features[18:27].eval())
        blocks.append(vgg.features[27:36].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, input, target):
        loss_perc = 0.0
        x = (input + 1) / 2
        y = (target + 1) / 2
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss_perc += torch.nn.functional.l1_loss(x, y)
        act_x = x.reshape(x.shape[0], x.shape[1], -1)
        act_y = y.reshape(y.shape[0], y.shape[1], -1)
        gram_x = act_x @ act_x.permute(0, 2, 1)
        gram_y = act_y @ act_y.permute(0, 2, 1)
        loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss_perc, loss_style