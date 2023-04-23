from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
import torch.nn.utils.prune as prune
import copy

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchinfo

import os
import argparse

from models.efficientnet import *
from models.resnet import *
from utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--summary', dest='summary', action='store_true',
                    help='show net summary ')
parser.add_argument('--dont_display_progress_bar', '-no_bar', action='store_true',
                    help='dont show progress bar')


parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')

parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--distillation', '-dist', action='store_true', help='apply distillation when training')


parser.add_argument('--skip_training', '-skip', action='store_true', help='skip initial training of the network')

parser.add_argument('--pruning_ratio', '-pr', default=0, type=float, help='pruning ratio')

parser.add_argument('--simple_pruning', '-prune', action='store_true', help='type of pruning')

parser.add_argument('--structured_pruning', '-st_pruning', action='store_true', help='type of pruning')

parser.add_argument('--iterative_pruning', '-iter_p', action='store_true', help='type of pruning')

parser.add_argument('--prune_first_layers', '-pfirst', action='store_true', help='in simple pruning it allows to only prune the first layers')

parser.add_argument('--fine_tune_pruning', '-ft', action='store_true', help='in simple pruning it allows fine tuning of the model')


parser.add_argument('--load_pruned_model', '-load_p', action='store_true', help='loads pruned model')

parser.add_argument('--sparsity', '-sp', action='store_true', help='shows sparsity information')




args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    # transforms.RandomRotation(60),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'


def save_net(snet,name = './saved/net.pth', acc=0, epoch=0):
    print('Saving..')
    state = {'net': snet.state_dict(), 'acc': acc, 'epoch': epoch,}
    torch.save(state, name)
    
def save_checkpoint(ckpt = './checkpoint/ckpt.pth', acc=0, epoch=0 ):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'lr': get_lr(optimizer),
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, ckpt)

def load_checkpoint(ckpt = './checkpoint/ckpt.pth'):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ckpt)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    args.lr = checkpoint['lr']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']

def load_teacher_checkpoint(ckpt = './checkpoint/teacher_ckpt.pth'):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ckpt)
    teacher_net.load_state_dict(checkpoint['net'])


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch, '\t Learning Rate: %.7f'  %get_lr(optimizer), '\t Best Acc: %.3f' %best_acc)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        
        if args.half:
            inputs = inputs.half()

        r = np.random.rand(1)        
        if args.beta > 0 and r < args.cutmix_prob: 
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            target_a = targets
            target_b = targets[rand_index]


            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

            outputs = net(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        
        else:
            inputs_ = (inputs[0].to(device='cpu').numpy())

            outputs = net(inputs)

            if args.distillation:
                teacher_outputs = teacher_net(inputs)
                distillation_loss = dist_loss(teacher_outputs, outputs)
            
                loss = criterion(outputs, targets) + distillation_loss

            else:
                loss = criterion(outputs, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs.float()
        loss = loss.float()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.dont_display_progress_bar==False:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, save_ckpt=True):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if args.half:
                inputs = inputs.half()
            outputs = net(inputs) 
            loss = criterion(outputs, targets)

            outputs = outputs.float()
            loss = loss.float()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.dont_display_progress_bar==False:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and save_ckpt==True:
        save_checkpoint(acc=acc, epoch=epoch)
        best_acc = acc

    return acc





if __name__ == "__main__":
   
    c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
    c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

    trainloader = DataLoader(c10train,batch_size=64,shuffle=True)
    testloader = DataLoader(c10test,batch_size=64) 


    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')

    # net = ModifiedEfficientNetB08()
    net = TinyEfficientNet()
    # net = TinyEfficientNetV2()


    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 250], gamma=0.1)

    if args.half:
        net.half()
        criterion.half()


    if args.resume:
        # Load checkpoint.
        load_checkpoint(ckpt='./checkpoint/ckpt.pth')
        edit_lr(optimizer, args.lr)


    if args.distillation:
        teacher_net = ResNet18()
        load_teacher_checkpoint(ckpt='./checkpoint/teacher_ckpt.pth')
        if args.half:
            teacher_net.half()
        teacher_net = teacher_net.to(device)


    original_size=torchinfo.summary(net, verbose=0).total_params
    if args.summary:
        torchinfo.summary(net)

    # test(start_epoch, save_ckpt=False)
    if args.skip_training==False:
        test(start_epoch)
        for epoch in range(start_epoch, start_epoch+300):
            train(epoch)
            test(epoch)
            scheduler.step()


    list_ = [       'layers.0',
                    'layers.1',
                    'layers.2',
                    'layers.3',
                    'layers.4',
                    'layers.5',
                    'layers.6',
                    'layers.7',
                    'layers.8',
                    'layers.9',
                    'layers.10',
                    'layers.11',
                    'layers.12',
                    'layers.13',
                    'layers.14',
                    'layers.15',   
                    'layers.16',   
                    'layers.17',   
                    'layers.18',   
                    'layers.19',   
                    'layers.20',   
            ]
 


    if args.pruning_ratio>0:
        parameters_to_prune = []
        for i, m in enumerate(net.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                aux = ((m, 'weight'))
                parameters_to_prune.insert(i, aux)

        prune_ratio = args.pruning_ratio

        if args.half:
            net.half()

        if args.structured_pruning:
            for i, m in enumerate(net.modules()):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    prune.ln_structured(m, name='weight',amount=prune_ratio/100, dim=1, n=float('-inf'))

        if args.iterative_pruning:

            for epoch in range(start_epoch, start_epoch+150):        

                if (epoch-start_epoch)%3==0:
                    prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=prune_ratio/100,
                    )
                    print('Current pruning ratio: ', 100-100*(1-prune_ratio/100)**((epoch-start_epoch+3)/3), '%')
                    print( torchinfo.summary(net, verbose=0).total_params)

                train(epoch)
                acc = test(epoch, save_ckpt=False)
                if args.dont_display_progress_bar:
                    print("Accuracy:", acc)
                
                if acc>=90:
                    last_prune_ratio = 100-100*(1-prune_ratio/100)**((epoch-start_epoch+3)/3)
                    for i, m in enumerate(net.modules()):
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                            prune.remove(m, 'weight')                   

                    save_checkpoint(ckpt='./checkpoint/ckpt_pruned_model.pth')
                    prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=last_prune_ratio/100,
                    )
                    current_size=torchinfo.summary(net, verbose=0).total_params

            print( torchinfo.summary(net, verbose=0).total_params )
            for i, m in enumerate(net.modules()):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    prune.remove(m, 'weight')

            load_checkpoint(ckpt='./checkpoint/ckpt_pruned_model.pth')
            prune_ratio= last_prune_ratio
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_ratio/100,
            )


        if args.simple_pruning:

            if args.prune_first_layers:
                i=0
                parameters_to_prune.clear()
                for name, layer in net.named_modules():
                    if name == 'layers.5':
                        break
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        aux = ((layer, 'weight'))
                        parameters_to_prune.insert(i, aux)
                        i+=1

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_ratio/100,
            )
            
            print("Fine-Tuning...")
            if args.fine_tune_pruning:
                for epoch in range(start_epoch, start_epoch+100):
                        train(epoch )
                        acc = test(epoch, save_ckpt=False)
                        if acc>90:
                            break

            current_size=torchinfo.summary(net, verbose=0).total_params
            print( current_size)


        if args.load_pruned_model:

            load_checkpoint(ckpt='./checkpoint/ckpt_pruned_model.pth')
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_ratio/100,
            )
            current_size=torchinfo.summary(net, verbose=0).total_params
            print( current_size)


        if args.sparsity:
            module_counter=0
            deleted_counter=0
            for name, layer in net.named_modules():
                
                if name in list_:
                    print(name)
                
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    n_layer_params = layer.weight.view(-1).size(0)
                    sparsity = 100. * float(torch.sum(layer.weight == 0)) / float(n_layer_params)                

                    module_counter+=1
                    if sparsity==100:
                        deleted_counter+=1
                    print("\t Sparsity:  {:.2f}%".format(sparsity), "\t Number of params:  ", n_layer_params,"\t", name)
                    
            print("Total number of modules:", module_counter)
            print("Number of deleted modules:", deleted_counter)
            print("Total sparsity:",  100*(1-current_size/original_size), "%")


        print("Pruned", prune_ratio, "%")
        acc=test(0, save_ckpt=False)
        print( acc )


        for i, m in enumerate(parameters_to_prune):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                prune.remove(m, 'weight')

        save_net(net, name='./saved/tinyeffnet.pth', acc=acc)





