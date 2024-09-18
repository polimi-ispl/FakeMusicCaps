import argparse
import data_lib
import network_models_lib
import torch
import os
import params
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl

# Use LaTeX for text rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']  # or any other serif font you like
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Continue with your plotting code
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


parser = argparse.ArgumentParser(description='CSCLassification')
parser.add_argument('--gpu', type=str, help='gpu', default='0')
parser.add_argument('--model_name', type=str, default='SpecResNet')
parser.add_argument('--audio_duration', type=float, help='Length of the audio slice in seconds',
                    default=10)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model selection
print('Closed set considering model {}'.format(args.model_name))
for args.audio_duration in [10, 7.5, 5, 2.5]:
    print('Audio duration {}'.format(args.audio_duration))
    # Model selection
    if args.model_name == 'M5':
        model = network_models_lib.M5(n_input=1, n_output=len(data_lib.model_labels))
        lr = 0.001
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
        model = network_models_lib.ResNet(img_channels=1, num_layers=18, block=network_models_lib.BasicBlock,
                                          num_classes=len(data_lib.model_labels))
        feat_type = 'freq'

    model.load_state_dict(torch.load( os.path.join(params.PARENT_DIR,'models','{}_duration_{}_secs.pth'.format(args.model_name, round(args.audio_duration,1)))))
    model.to(device)

    test_closed_data = data_lib.MusicDeepFakeDataset(data_lib.test_files, data_lib.model_labels,
                                                     args.audio_duration,feat_type=feat_type)  # ERROR

    test_closed_dataloader = torch.utils.data.DataLoader(test_closed_data, batch_size=1, shuffle=True, num_workers=8)


                                                         #### CLOSED SET ANALYSIS ##############################################################################################
    model.eval()
    correct = 0
    pred_list = []
    target_list = []
    for data, target in test_closed_dataloader:
        data = data.to(device)
        target = target.to(device)
        # apply transform and model on whole batch directly on device
        output = model(data)

        # For Accuracy
        pred = get_likely_index(output)
        correct += number_of_correct(pred.squeeze(), target.squeeze())

        # For confusion matrix
        pred_list = pred_list + pred.cpu().numpy()[:, 0].tolist()
        target_list = target_list +target.cpu().to(torch.int64).numpy()[:, 0].tolist()
        # update progress bar
        #pbar.update(pbar_update)

    cm = confusion_matrix(target_list, pred_list, normalize='true')
    np.save(os.path.join(params.PARENT_DIR,'figures/cm_closed_set_{}_{}_sec.npy'.format(args.model_name,args.audio_duration)),cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[r'REAL',r'TTM01', r'TTM02', r'TTM03', r'TTM04', r'TTM05'])
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    # Adjust global font sizes

    #plt.figure()
    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues,colorbar=False)
    # Customize tick labels and axis labels to use LaTeX
    plt.xticks(rotation=45)
    plt.xlabel(r'Predicted Labels',fontsize=15)
    plt.ylabel(r'True Labels',fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(params.PARENT_DIR,'figures/cm_closed_set_{}_{}_sec.png'.format(args.model_name,args.audio_duration),dpi=300))
    plt.show()
    # Balanced accuracy score
    ACC_B = balanced_accuracy_score(target_list, pred_list)

    # Precision
    precision_classes = np.zeros(len(data_lib.models_names))
    occurrence_classes = np.zeros(len(data_lib.models_names))

    for idx in range(len(pred_list)):
        occurrence_classes[pred_list[idx]] += 1
        if pred_list[idx] == target_list[idx]:
            precision_classes[target_list[idx]] += 1
    precision_classes/=occurrence_classes
    precision_tot = np.mean(precision_classes)


    # Recall
    recall_classes = np.zeros(len(data_lib.models_names))
    occurrence_classes_recall = np.zeros(len(data_lib.models_names))

    for idx in range(len(pred_list)):
        occurrence_classes_recall[target_list[idx]] += 1
        if pred_list[idx] == target_list[idx]:
            recall_classes[target_list[idx]] += 1
    recall_classes /= occurrence_classes_recall
    recall_tot = np.mean(recall_classes)

    # F1-score
    F1_per_class = f1_score(target_list, pred_list, average=None)
    F1_avg = f1_score(target_list, pred_list, average='macro')


    results = np.array([round(ACC_B, 2), round(precision_tot, 2), round(recall_tot, 2),  round(F1_avg, 2)])
    results_filename = os.path.join(params.PARENT_DIR,'results','closed_set_{}_{}_sec.npy'.format(args.model_name,
                                                                                                  args.audio_duration))
    np.save(results_filename, results)


    print('ACC_B: {} Precision {} Recall {} F1 Score {}'.format(round(ACC_B, 2), round(precision_tot, 2),
                                                                round(recall_tot, 2),  round(F1_avg, 2)))

