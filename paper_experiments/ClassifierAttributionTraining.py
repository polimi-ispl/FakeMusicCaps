import torch
import argparse
import os
import sys
sys.path.append('../')
import torch.nn.functional as F
import data_lib
import network_models_lib
import datetime
import params
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser(description='TTM attribution training')
parser.add_argument('--gpu', type=str, help='gpu', default='4')
parser.add_argument('--model_name', type=str, default='SpecResNet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--log_dir', type=str, help='store tensorboard info',
                    default='')

parser.add_argument('--audio_duration', type=float, help='Length of the audio slice in seconds',
                    default=7.5)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implement Early Stopping


# Model selection
if args.model_name == 'M5':
    model = network_models_lib.M5(n_input=1, n_output=len(data_lib.model_labels))
    lr = 0.0001
    feat_type = 'raw'
elif args.model_name == 'RawNet2':
    d_args = {"nb_samp": int(args.audio_duration * params.DESIRED_SR), "first_conv": 3, "in_channels": 1,
              "filts": [128, [128, 128], [128, 256], [256, 256]],
              "blocks": [2, 4], "nb_fc_node": 1024, "gru_node": 1024, "nb_gru_layer": 1,
              "nb_classes": len(data_lib.model_labels)}
    lr = 0.0001
    print('USING MODEL {}'.format(args.model_name))
    model = network_models_lib.RawNet2(d_args)
    feat_type = 'raw'
elif args.model_name == 'SpecResNet':
    lr = 0.0001
    print('USING MODEL {}'.format(args.model_name))
    model = network_models_lib.ResNet(img_channels=1, num_layers=18, block=network_models_lib.BasicBlock, num_classes=len(data_lib.model_labels))
    feat_type ='freq'

# Data Loading
num_workers = 16
training_data = data_lib.MusicDeepFakeDataset(data_lib.train_set, data_lib.model_labels, args.audio_duration,feat_type=feat_type)
val_data = data_lib.MusicDeepFakeDataset(data_lib.val_set, data_lib.model_labels, args.audio_duration,feat_type=feat_type)
test_closed_data = data_lib.MusicDeepFakeDataset(data_lib.test_files, data_lib.model_labels,
                                                 args.audio_duration, feat_type=feat_type)  # ERROR
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
test_closed_dataloader = torch.utils.data.DataLoader(test_closed_data, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=num_workers)

def train(model, epoch, log_interval,optimizer):
    model.train()
    losses = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target.squeeze().to(torch.int64))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} ({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}")

        # record loss
        losses = losses + loss.item()
    losses /= len(train_dataloader)
    return losses


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def validation(model, epoch):
    model.eval()
    correct = 0
    losses_val = 0
    for data, target in val_dataloader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred.squeeze(), target.squeeze())

        loss = F.nll_loss(output.squeeze(), target.squeeze().to(torch.int64))


        # record loss
        losses_val = losses_val + loss.item()
    losses_val /= len(val_dataloader)
    accuracy = correct/len(val_dataloader.dataset)
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_dataloader.dataset)} ({100. * correct / len(val_dataloader.dataset):.0f}%)\n")
    return losses_val, accuracy

#### OPEN SET ANALYSIS ##############################################################################################
def get_likely_index_openset(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


# Open SummaryWriter for Tensorboard
current_time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=os.path.join(params.LOG_DIR, args.model_name +'_'+current_time ))



def main():

    model.to(device)
    print("Using model_{}_{}_secs".format(args.model_name, round(args.audio_duration,1)))

    n = count_parameters(model)
    print("Number of parameters: %s" % n)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10


    # The transform needs to live on the same device as the model and the data.
    VAL_LOSS_MIN = 100000000000
    log_interval = 1000
    n_epoch = 100
    early_stop_cnt = 0
    patience = 10

    for epoch in tqdm(range(1, n_epoch + 1)):
        train_loss = train(model, epoch, log_interval, optimizer)
        val_loss, accuracy_epoch = validation(model, epoch)
        writer.add_scalar('Loss/train',  train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        scheduler.step()

        if val_loss < VAL_LOSS_MIN:
            VAL_LOSS_MIN = val_loss
            print('ok Saving model epoch {}'.format(epoch))
            torch.save(model.state_dict(), os.path.join(params.PARENT_DIR,'models','{}_duration_{}_secs.pth'.format(args.model_name, round(args.audio_duration,1))))
            early_stop_cnt = 0
        else:
            early_stop_cnt = early_stop_cnt + 1
            print('PATIENCE {}/{}'.format(early_stop_cnt, patience))

        if early_stop_cnt == patience:
            break



if __name__ == '__main__':
    main()