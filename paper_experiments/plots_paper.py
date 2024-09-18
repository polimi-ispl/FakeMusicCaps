import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('../')
plt.rcParams['text.usetex'] = True
from sklearn.metrics import ConfusionMatrixDisplay
RESULTS_DIR = os.path.join(params.PARENT_PATH,'results')
FIGURES_DIR = os.path.join(params.PARENT_PATH,'figures')
time_windows = [10, 7.5, 5, 2.5]
model_names = ['M5', 'RawNet2', 'SpecResNet']

# PLOT IMAGES (ACC VS TIME) - Closed Set
colors = ['#1F77B4','#AEC7E8','#17BECF','#9EDAE5']
figsize=(5,2)
fontsize=20
linewidth = 2
plt.figure(figsize=figsize)
for i_m, m_n in enumerate(model_names):
    ACC_B = np.zeros(len(time_windows))
    for i_t, t_w in enumerate(time_windows):
        filename ='closed_set_{}_{}_sec.npy'.format(m_n,t_w)
        arr = np.load(os.path.join(RESULTS_DIR,filename))
        ACC_B[i_t] = arr[0]
    plt.plot(time_windows,ACC_B,'s-',linewidth=linewidth,color=colors[i_m])
plt.grid('major')
plt.xlabel(r'$\mathrm{Window}~\mathrm{length}~\mathrm{[s]}$', fontsize=fontsize)
plt.ylabel(r'$\mathrm{ACC_B}$', fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'closed_set_vs_window.png', dpi=300))

plt.show()

# PLOT IMAGES (ACC VS TIME) - Open Set Threshold
plt.figure(figsize=figsize)
for i_m, m_n in enumerate(model_names):
    ACC_B = np.zeros(len(time_windows))
    for i_t, t_w in enumerate(time_windows):
        filename ='open_set_thresh__{}_{}_sec.npy'.format(m_n,t_w)
        arr = np.load(os.path.join(RESULTS_DIR,filename))
        ACC_B[i_t] = arr[0]
    plt.plot(time_windows,ACC_B,'s-',linewidth=linewidth,color=colors[i_m])
plt.grid('major')
plt.xlabel(r'$\mathrm{Window}~\mathrm{length}~\mathrm{[s]}$', fontsize=fontsize)
plt.ylabel(r'$\mathrm{ACC_B}$', fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'open_set_thresh_vs_window.png', dpi=300))

plt.show()

# PLOT IMAGES (ACC VS TIME) - Open Set SVM
plt.figure(figsize=figsize)
for i_m, m_n in enumerate(model_names):
    ACC_B = np.zeros(len(time_windows))
    for i_t, t_w in enumerate(time_windows):
        filename ='open_set_svm__{}_{}_sec.npy'.format(m_n,t_w)
        arr = np.load(os.path.join(RESULTS_DIR,filename))
        ACC_B[i_t] = arr[0]
    plt.plot(time_windows,ACC_B,'s-',linewidth=linewidth,color=colors[i_m])
plt.grid('major')
plt.xlabel(r'$\mathrm{Window}~\mathrm{length}~\mathrm{[s]}$', fontsize=fontsize)
plt.ylabel(r'$\mathrm{ACC_B}$', fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'open_set_svm_vs_window.png', dpi=300))

plt.show()

# Print for table
t_w = time_windows[0]
print('Generating CLOSED SET table for time window {}'.format(t_w))
for i_m, m_n in enumerate(model_names):
    filename = 'closed_set_{}_{}_sec.npy'.format(m_n, t_w)
    arr = np.load(os.path.join(RESULTS_DIR, filename))
    print('{} &{} &{}&{}& {}\\'.format(m_n,arr[0],arr[1],arr[2],arr[3]))

print('Generating OPEN SET THRESH table for time window {}'.format(t_w))
for i_m, m_n in enumerate(model_names):
    filename = 'open_set_thresh__{}_{}_sec.npy'.format(m_n, t_w)
    arr = np.load(os.path.join(RESULTS_DIR, filename))
    print('{} &{} &{}&{}& {}\\'.format(m_n, arr[0], arr[1], arr[2], arr[3]))


print('Generating OPEN SET SVM table for time window {}'.format(t_w))
for i_m, m_n in enumerate(model_names):
    filename = 'open_set_svm__{}_{}_sec.npy'.format(m_n, t_w)
    arr = np.load(os.path.join(RESULTS_DIR, filename))
    print('{} &{} &{}&{}& {}\\'.format(m_n, arr[0], arr[1], arr[2], arr[3]))


# PLOT CONFUSION MATRICES
FONTSIZE = 15
# Print for table
t_w = time_windows[0]
print('Generating CLOSED SET table for time window {}'.format(t_w))
for i_m, m_n in enumerate(model_names):
    cm = np.load(os.path.join(FIGURES_DIR,'cm_closed_set_{}_{}_sec.npy'.format(m_n, t_w)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[r'REAL',r'TTM01', r'TTM02', r'TTM03', r'TTM04', r'TTM05'])
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    # Adjust global font sizes

    #plt.figure()
    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues,colorbar=False)
    # Customize tick labels and axis labels to use LaTeX
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks( fontsize=12)
    plt.xlabel(r'Predicted Labels',fontsize=FONTSIZE)
    plt.ylabel(r'True Labels',fontsize=FONTSIZE)
    plt.tight_layout()
    for labels in disp.text_.ravel():
        labels.set_fontsize(15)
    plt.savefig(os.path.join(FIGURES_DIR,'cm_closed_set_{}_{}_sec.png'.format(m_n, t_w)),dpi=300)
    plt.show()

print('Generating OPEN SET THRESH table for time window {}'.format(t_w))
for i_m, m_n in enumerate(model_names):
    cm = np.load(os.path.join(FIGURES_DIR,'cm_open_set_thresh_{}_{}_sec.npy'.format(m_n, t_w)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[r'REAL', r'TTM01', r'TTM02', r'TTM03', r'TTM04', r'TTM05',r'UNKWN'])
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    # Adjust global font sizes

    #plt.figure()
    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues,colorbar=False)
    # Customize tick labels and axis labels to use LaTeX
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks( fontsize=12)
    plt.xlabel(r'Predicted Labels',fontsize=FONTSIZE)
    plt.ylabel(r'True Labels',fontsize=FONTSIZE)
    plt.tight_layout()
    for labels in disp.text_.ravel():
        labels.set_fontsize(15)
    plt.savefig(os.path.join(FIGURES_DIR,'cm_open_set_thresh_{}_{}_sec.png'.format(m_n, t_w)),dpi=300)
    plt.show()


print('Generating OPEN SET SVM table for time window {}'.format(t_w))
for i_m, m_n in enumerate(model_names):
    cm = np.load(os.path.join(FIGURES_DIR,'cm_open_set_svm_{}_{}_sec.npy'.format(m_n, t_w)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[r'REAL', r'TTM01', r'TTM02', r'TTM03', r'TTM04', r'TTM05',r'UNKWN'])
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    # Adjust global font sizes
    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues,colorbar=False)
    # Customize tick labels and axis labels to use LaTeX
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=12)
    plt.xlabel(r'Predicted Labels',fontsize=FONTSIZE)
    plt.ylabel(r'True Labels',fontsize=FONTSIZE)
    plt.tight_layout()
    for labels in disp.text_.ravel():
        labels.set_fontsize(15)
    plt.savefig(os.path.join(FIGURES_DIR,'cm_open_set_svm_{}_{}_sec.png'.format(m_n, t_w)),dpi=300)
    plt.show()