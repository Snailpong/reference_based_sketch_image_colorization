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

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

    def forward(self, input, target):
        loss_perc = 0.0
        x = (input + 1) / 2
        y = (target + 1) / 2
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss_perc += torch.nn.functional.l1_loss(x, y)
        gram_x = self.gram_matrix(x)
        gram_y = self.gram_matrix(y)
        loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss_perc, loss_style


def similarity_based_triplet_loss(dots, margin):
    loss = torch.max(-dots[0] + dots[1] + margin, torch.zeros_like(dots[0]))
    return torch.mean(loss)