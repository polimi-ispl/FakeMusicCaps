import sys
sys.path.append('../')
import data_lib
import os
import torch
import network_models_lib
import argparse
import copy
import matplotlib.pyplot as plt
import params
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.svm import OneClassSVM
from tqdm import tqdm
"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 15})
"""
from data_lib import SUNOCAPS_PATH, model_labels, MusicDeepFakeDataset, test_suno
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

parser = argparse.ArgumentParser(description='OSCLassification')
parser.add_argument('--gpu', type=str, help='gpu', default='1')
parser.add_argument('--model_name', type=str, default='SpecResNet')
parser.add_argument('--log_dir', type=str, help='store tensorboard info',
                    default='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/logs')
parser.add_argument('--audio_duration', type=float, help='Length of the audio slice in seconds',
                    default=10)
args = parser.parse_args()

for model in ['M5', 'RawNet2','SpecResNet']:
    print('Open set (SVM) considering model {}'.format(args.model_name))
    for args.audio_duration in [10, 7.5, 5, 2.5]:
        # Model selection
        # Model selection
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        OpenClassLabel = 6
        test_suno_files = [os.path.join(SUNOCAPS_PATH,path) for path in test_suno]
        model_labels_open_set = copy.deepcopy(model_labels)
        model_labels_open_set.update({'SunoCaps': OpenClassLabel})
        test_open_data = MusicDeepFakeDataset(test_suno_files+data_lib.test_files, model_labels_open_set,args.audio_duration,feat_type=feat_type) # ERROR
        test_open_dataloader = torch.utils.data.DataLoader(test_open_data, batch_size=1, shuffle=True,num_workers=0)

        X_train = [] # s= [[0], [0.44], [0.45], [0.46], [1]]
        # Create training set
        training_data = data_lib.MusicDeepFakeDataset(data_lib.train_set, data_lib.model_labels, args.audio_duration,feat_type=feat_type)
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True, num_workers=8)

        print('Creating train set for SVM')
        for data, target in tqdm(train_dataloader):
            data = data.to(device)
            output = model(data)
            X_train += output[:, 0, :].tolist()

        # Train One-class support-vector-machine
        print('Start SVM training')
        #clf = OneClassSVM(kernel='poly',gamma='auto',nu=0.8).fit(X_train)

        clf = OneClassSVM(gamma='auto').fit(X_train)

        print('SVM Training ended')
        #https: // scikit - learn.org / stable / modules / outlier_detection.html  # outlier-detection

        print('Start open set classification')
        # SVM + CLASSIFICATION
        correct = 0
        pred_list = []
        target_list = []
        for data, target in test_open_dataloader:
            data = data.to(device)
            target = target.to(device)
            # apply transform and model on whole batch directly on device
            output = model(data)
            outlier = clf.predict(output[:, 0, :].tolist())[0]
            #print('outlier: {} target: {}'.format(outlier,target))
            # For Accuracy
            pred = output.argmax(dim=1)

            # If an outlier is detected assign Unkown Class
            if outlier == -1:
                pred = torch.Tensor([[OpenClassLabel]])

            correct += number_of_correct(pred.squeeze(), target.squeeze())


            # For confusion matrix
            pred_list = pred_list + pred.cpu().numpy()[:, 0].tolist()
            target_list = target_list +target.cpu().to(torch.int64).numpy()[:, 0].tolist()
            # update progress bar
            #pbar.update(pbar_update)

        cm = confusion_matrix(target_list, pred_list, normalize='true')
        np.save(os.path.join(params.PARENT_DIR,'figures/cm_open_set_svm_{}_{}_sec.npy'.format(args.model_name,args.audio_duration)),cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_lib.models_names + ['SunoCaps/Unknown'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=[r'REAL', r'TTM01', r'TTM02', r'TTM03', r'TTM04', r'TTM05',r'UNKWN'])
        # Enable LaTeX rendering
        plt.rcParams['text.usetex'] = True
        # Adjust global font sizes
        # Plot confusion matrix
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        # Customize tick labels and axis labels to use LaTeX
        plt.xticks(rotation=45)
        plt.xlabel(r'Predicted Labels', fontsize=15)
        plt.ylabel(r'True Labels', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(params.PARENT_DIR,'figures/cm_open_set_svm_{}_{}_sec.png'.format(args.model_name,args.audio_duration)),dpi=300)

        plt.show()

        # Balanced accuracy score
        ACC_B = balanced_accuracy_score(target_list, pred_list)

        # Precision
        precision_classes = np.zeros(len(data_lib.models_names)+1)
        occurrence_classes = np.zeros(len(data_lib.models_names)+1)

        for idx in range(len(pred_list)):
            occurrence_classes[int(pred_list[idx])] += 1
            if int(pred_list[idx]) == int(target_list[idx]):
                precision_classes[int(target_list[idx])] += 1
        precision_classes /= occurrence_classes
        precision_tot = np.mean(precision_classes)



        # Recall
        recall_classes = np.zeros(len(data_lib.models_names)+1)
        occurrence_classes_recall = np.zeros(len(data_lib.models_names)+1)

        for idx in range(len(pred_list)):
            occurrence_classes_recall[int(target_list[idx])] += 1
            if int(pred_list[idx]) == int(target_list[idx]):
                recall_classes[int(target_list[idx])] += 1
        recall_classes /= occurrence_classes_recall
        recall_tot = np.mean(recall_classes)

        # F1-score
        F1_per_class = f1_score(target_list, pred_list, average=None)
        F1_avg = f1_score(target_list, pred_list, average='macro')

        # AUC
        # from sklearn.metrics import roc_auc_score
        # AUC = roc_auc_score

        results = np.array([round(ACC_B, 2), round(precision_tot, 2), round(recall_tot, 2),  round(F1_avg, 2)])
        results_filename = os.path.join(params.PARENT_DIR,'results','open_set_svm__{}_{}_sec.npy'.format(args.model_name,
                                                                                                  args.audio_duration))


        np.save(results_filename, results)

        print('ACC_B: {} Precision {} Recall {}'
              ' F1 Score {}'.format(round(ACC_B, 2),
                                    round(precision_tot, 2),
                                    round(recall_tot, 2),
                                    round(F1_avg, 2)))




