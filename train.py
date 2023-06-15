# -*- coding:utf-8 -*-
import os
import argparse
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from sklearn.preprocessing import label_binarize
import sklearn.metrics as skmetrics
import numpy as np
import torch
import time
import json
import logging
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as mo
import timm
import tools.tensorb as ts
import model.rexnetv1 as rex
import model.AGMB as agmb
import model.AGMBv2 as agmb2 #agmb + scn conv
import model.AGMBv25 as agmb25 #agmb + scn conv + 3 block to 1
import model.AGMBv3 as agmb3 #agmb + scn conv +transgf(3 block to 1)
import model.TransFG as transfg
import model.botnet as bot


from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import med_dataset
from medmnist.load_data import MyDataset
from medmnist.Bdataset import B_dataset
from tools.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO
from medmnist.BiRAmodel import KeNet


def main(configs_path , output_root ,results_path):
    ''' main function
    :param flag: name of subset

    '''

    # configs reading
    configs_file_path = os.path.join(configs_path, args.configs_name)
    with open(configs_file_path, 'r') as f:
        cnn = json.load(f)


    with open(os.path.join(configs_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)


    #task = 'multi-label, multi-class'
    #task = 'multi-label, binary-class'
    task = 'multi-class'
    n_channels = cnn['n_channels']   #this label the level 0,1,2,3,4
    n_classes = cnn['n_classes']  #how much label class

    val_auc_list = []
    dir_path = os.path.join(results_path, '%s_checkpoints' )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('==> Preparing data...')
    
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])
         
    
    """train_dataset = med_dataset(img_path=cnn['data_path_train'],      # a dataset object
                                label_path=cnn['csv_path_train'],
                                transform=train_transform
                                    )
    train_loader = data.DataLoader(dataset=train_dataset,  # a dataloader object
                                   batch_size=cnn['batch_size'],
                                   shuffle=True)
    val_dataset   = med_dataset(img_path=cnn['data_path_valid'],
                                label_path=cnn['csv_path_val'],
                                transform=val_transform
                                    )
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=cnn['batch_size'],
                                 shuffle=True)
    test_dataset = med_dataset(img_path=cnn['data_path_test'],
                                label_path=cnn['csv_path_test'],
                                transform=test_transform
                                    )
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=cnn['batch_size'],
                                  shuffle=True)"""


    train_dataset = B_dataset(img_path=cnn['data_path_train'],      # a dataset object
                                label_path=cnn['csv_path_train'],
                                transform=train_transform
                                    )
    train_loader = data.DataLoader(dataset=train_dataset,  # a dataloader object
                                   batch_size=cnn['batch_size'],
                                   shuffle=True)
    val_dataset   = B_dataset(img_path=cnn['data_path_valid'],
                                label_path=cnn['csv_path_val'],
                                transform=val_transform
                                    )
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=cnn['batch_size'],
                                 shuffle=True)
    test_dataset = B_dataset(img_path=cnn['data_path_test'],
                                label_path=cnn['csv_path_test'],
                                transform=test_transform
                                    )
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=cnn['batch_size'],
                                  shuffle=True)

    print('==> Building and training model...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if cnn['model'] == 'ResNet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
    elif cnn['model'] == 'ResNet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
    elif cnn['model'] == 'EfficientNet':
        model = timm.create_model('efficientnet_b0', num_classes=n_classes, pretrained = False)
    elif cnn['model'] == 'MobileNet3':
        model = timm.create_model('mobilenetv3_large_100', num_classes=n_classes, pretrained = False)
    elif cnn['model'] == 'MobileNetV2':
        model = timm.create_model('mobilenetv2_100', num_classes=n_classes, pretrained = False)
    elif cnn['model'] == 'RS18':
        model = mo.resnet18(pretrained = False , progress = True ,num_classes = n_classes)#num_classes is use to determine how many situations each label has
    elif cnn['model'] == 'RSV1':
        model = rex.ReXNetV1(width_mult=1.0).cuda()
    elif cnn['model'] == 'AGMB':
        model = agmb.AGGT50(n_classes)
    elif cnn['model'] == 'AGMB2':
        model = agmb2.AGGT50(n_classes)
    elif cnn['model'] == 'AGMB25':
        model = agmb25.AGGT50(n_classes)
    elif cnn['model'] == 'AGMB3':
        model = agmb3.AGGT50(n_classes)
    elif cnn['model'] == 'TransFG':
        configT = transfg.CONFIGS["ViT-B_16"]
        configT.split = 'non-overlap'
        configT.slide_step = 12
        model = transfg.VisionTransformer(configT, 128, zero_head=True, num_classes=n_classes,smoothing_value=0)
    elif cnn['model'] == 'bot':
        model = bot.botnet50(n_classes)
    elif cnn['model'] == 'BIRANET':
        model = KeNet(n_classes).cuda()
        if path:
         model.load_state_dict(torch.load(path))
        unchanged_params = list(map(id, model.features[:-2].parameters()))
        unchanged_params += list(map(id, model.up_c2.parameters()))
        training_params = filter(lambda p: id(p) not in unchanged_params, model.parameters())
        for param in model.up_c2.parameters():
         param.requires_grad = False
        for param in model.features[:-2].parameters():
         param.requires_grad = False
    else:
        raise Exception("I have not add any models. ")

    model.cuda()
    print(torch.cuda.is_available())
    print(torch.cuda.device_count(),"GPUS!")
    #model = nn.DataParallel(model)

    if cnn['n_classes'] == 2:
        #criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()
        task = 'binary-class'
        print(task)
    elif cnn['n_classes'] == 3:
        criterion = nn.CrossEntropyLoss()
        task = 'multi-class3'
        print(task)
    elif cnn['n_classes'] == 5:
        criterion = nn.CrossEntropyLoss()
        task = 'multi-class'
        print(task)
    elif cnn['n_classes'] == 6:
        criterion = nn.CrossEntropyLoss()
        task = 'multi-class'
        print(task)
    
    optimizer = optim.SGD(model.parameters(), cnn['lr'], momentum=0.9)
    ExpLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,0.0005)
    train_start_time = time.time()
    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(results_path)
    loss_valid_best = float('inf')
    acc_valid_best = 0
    sen_valid_best = 0

    for epoch in trange(cnn['start_epoch'], cnn['end_epoch']):
        summary_train = train(model, optimizer,criterion, train_loader, device, task ,summary_train, train_loader)
        summary_valid = val(model, val_loader, device, val_auc_list, task, dir_path, epoch , summary_valid , criterion ,val_loader,criterion)
        print(epoch, end=',')
        print("learning rate is ", optimizer.param_groups[0]["lr"])
        ExpLR.step()
        ts.scalar(model,summary_train,summary_valid,summary_writer,loss_valid_best,acc_valid_best,sen_valid_best,results_path)


    train_spent_time = time.time() - train_start_time
    with open(os.path.join(results_path, 'final_results.txt'), "a+") as f:
        f.write('Training time cost: ' + str(train_spent_time) + '\n')
    summary_writer.close()

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    """restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index])) #save the location
    model.load_state_dict(torch.load(restore_model_path)['net'])"""


    summary_test_train = {}
    summary_test_valid = {}
    summary_test_test = {}
    summary_test_train =test(model,'train',train_loader,device,task,
                             criterion, summary_test_train, results_path , output_root= output_root)
    summary_test_valid = test(model, 'val', val_loader, device,  task ,
                              criterion , summary_test_valid , results_path , output_root=output_root)
    summary_test_test = test(model,'test',test_loader,device,task,
                             criterion , summary_test_test , results_path , output_root=output_root )
    
    loss_test_best = float('inf')
    acc_test_best = 0
    sen_test_best = 0

    ts.testscalar (model, summary_train ,summary_test_test, summary_writer, loss_test_best, acc_test_best, sen_test_best, results_path)
    

    # wen dang li de dong xi jiu shi xie zai zhe li
    with open(os.path.join(results_path, 'final_results.txt'), "a+") as f:
        for key1 in summary_test_train:
            f.write(key1 + ': ' + str(summary_test_train[key1]) + '\n')
        for key2 in summary_test_valid:
            f.write(key2 + ': ' + str(summary_test_valid[key2]) + '\n')
        for key3 in summary_test_test:
            f.write(key3 + ': ' + str(summary_test_test[key3]) + '\n')



def train(model, optimizer,criterion, train_loader, device, task , summary , dataloader_train):

    model.train()
    loss_sum = 0
    probs_list = []
    predicts_list = []
    target_train_list = []
    steps = len(dataloader_train)


    for batch_idx, (inputs,targets) in enumerate(train_loader):
 
        outputs = model(inputs.to(device))
        #print(outputs)
        # probs is to predict the probability of each class,predict is use the probs to predict the class is 0 or 1 ,
        # target is to predict the data which class it is.

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()

        targets = targets.view(-1)
        loss.backward()
        optimizer.step()

        probs = outputs.sigmoid()
        #predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
        predicts = probs.type(torch.cuda.FloatTensor)
        #this three sentences just make the array to list.
        prob = probs.cpu().data.numpy().tolist()
        predict = predicts.cpu().data.numpy().tolist()
        target = targets.cpu().data.numpy().tolist()

        probs_list.extend(prob)
        predicts_list.extend(predict)
        target_train_list.extend(target)

        loss_data = loss.data
        summary['step'] += 1
        loss_sum += loss_data

  

    summary = ts.summarytrain(probs_list,predicts_list,target_train_list,task,summary,loss_data,loss_sum,steps)

    return summary


def val(model, val_loader, device, val_auc_list, task, dir_path, epoch , summary ,loss_fn ,dataloader_valid,criterion):

    model.eval()
    steps = len(dataloader_valid)
    loss_sum = 0
    probs_list = []
    predicts_list = []
    target_valid_list = []
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs,targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))
            probs = outputs.sigmoid()

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else :

                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
             
          
            #squeeze is use to convert an array representing a vector to an array of rank 1
            #one_hot = torch.nn.functional.one_hot(targets, num_classes=4)
 
            loss = criterion(outputs, targets)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
            
            loss_data = loss.data
            loss_sum += loss_data

            
            predicts = probs.type(torch.cuda.FloatTensor)
            probs = probs.cpu().data.numpy().tolist()
            predicts = predicts.cpu().data.numpy().tolist()
            target_valid = targets.cpu().data.numpy().tolist()
            

            probs_list.extend(probs)
            predicts_list.extend(predicts)
            target_valid_list.extend(target_valid)
            """print(target_valid_list)
            y_test_one_hot = label_binarize(target_valid_list, np.arange(3))
            y_test_one_hot = np.array(y_test_one_hot)
            print(y_test_one_hot)
            print(y_test_one_hot.shape)"""
            


        y_true = y_true.cpu().numpy()            #target_valid_array
        y_score = y_score.detach().cpu().numpy() #probs

        auc = getAUC(y_true,y_score,task)
        acc = getACC(y_true,y_score,task)
        print('this is val phase : val AUC: %.5f ACC: %.5f' % (auc, acc))
        summary = ts.summaryval(predicts_list,target_valid_list,task,summary,val_auc_list,loss_sum,steps,y_true,y_score)
        stage = 'val'
        ts.show_roc(y_score, y_true, results_path , task , stage)
        
    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    """path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)"""

    return summary


def test(model, split, data_loader, device, task , criterion, summary ,results_path,output_root=None):


    model.eval()
    steps = len(data_loader)
    loss_sum = 0
    probs_list = []
    predicts_list = []
    target_valid_list = []
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs,targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            probs = outputs.sigmoid()

            """if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            elif task == 'binary-class':
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1).to(device)"""

            if task == 'multi-class':
                boss = targets.long().to(device)
                loss = criterion(outputs, boss)
            elif task == 'multi-class3':
                boss = targets.long().to(device)
                loss = criterion(outputs, boss)
            elif task == 'binary-class':
                boss = targets.long().to(device)
                boss = boss.squeeze()
                loss = criterion(outputs, boss)
            
            y_true = y_true.to(device)
            targets = targets.to(device)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

            loss_data = loss.data
            loss_sum += loss_data

            predicts = probs.type(torch.cuda.FloatTensor)
            probs = probs.cpu().data.numpy().tolist()
            predicts = predicts.cpu().data.numpy().tolist()
            target_valid = targets.cpu().data.numpy().tolist()

            probs_list.extend(probs)
            predicts_list.extend(predicts)
            target_valid_list.extend(target_valid)

        y_true = y_true.cpu().numpy()  # target_valid_array
        y_score = y_score.detach().cpu().numpy()  # probs
        auc = getAUC(y_true,y_score,task)
        acc = getACC(y_true,y_score,task)
        print('this is test phase: %s AUC: %.5f ACC: %.5f' % (split, auc, acc))
        summary = ts.summarytest(predicts_list, target_valid_list, task, summary, split, loss_sum, steps, y_true,y_score)
        ts.show_confMat(predicts_list, target_valid_list, results_path , task)
        stage = 'test'
        ts.show_roc(predicts_list, target_valid_list, results_path , task ,stage)
        
        

        if output_root is not None:
            output_dir = os.path.join(output_root, 'CFP')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            #save_results

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--configs_name',
                        default=None,
                        metavar='CONFIG_NAME',
                        type=str,
                        help='Name of the config name that required')
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--result_path_name',
                        default='./result',
                        help='the tensorboard result saving path',
                        type=str)



    args = parser.parse_args()
    curr_path = os.getcwd()  # getcwd is to get the cuurent path
    configs_path = os.path.join(curr_path, 'configs')  # os.path.join is use to add 'configs' to current path
    output_root = args.output_root
    results_path = args.result_path_name


    main(configs_path,
         output_root,
         results_path)
