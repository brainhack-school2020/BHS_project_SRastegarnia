#!/usr/bin/env python
# coding: utf-8

# ## Haxby data set:
# Haxby is a high-quality block-design fMRI dataset from a study on face & object representation in the human ventral temporal cortex (This cortex is involved in the high-level visual processing of complex stimuli). It consists of 6 subjects with 12 runs per subject. In this experiment during each run, the subjects passively viewed greyscale images of 8 object categories, grouped in 24s blocks separated by rest periods. Each image was shown for 500ms and was followed by a 1500ms inter-stimulus interval.
# 
# ## Project Goal
# For this project i am trying machine learning and deep learning methods to learning about barin decoding and predicting which object category the subject saw by analyzing the fMRI activity recorded masks of the ventral stream.

# In[1]:


import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from nilearn.plotting import plot_anat, show, plot_stat_map, plot_matrix
from nilearn import datasets, plotting, image
from nilearn.image import mean_img, get_data 
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier,  RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


# In[2]:


#%matplotlib inline
#%load_ext memory_profiler


# ## Dataset

# In[3]:


# If we don't define which subject by default 2nd subject will be fetched.
haxby_ds = datasets.fetch_haxby(fetch_stimuli=True)
# print(haxby_ds)

# To define which subject to be fetched:
# haxby_ds = datasets.fetch_haxby(subjects=[], fetch_stimuli=True)

len(haxby_ds.func)


# In[4]:


# Look inside the data
haxby_ds.keys()


# In[5]:


mask_file = haxby_ds.mask
labels = haxby_ds.session_target[0]
mask_vt_file = haxby_ds.mask_vt[0]
mask_face_file = haxby_ds.mask_face[0]

# 'func' is a list of filenames: one for each subject
func_file = haxby_ds.func[0]

# Load the behavioral data that I will predict
beh_label = pd.read_csv(haxby_ds.session_target[0], sep=" ")

# Extract tags indicating to which acquisition run a tag belongs
session = beh_label['chunks']

# Preparing the data (Load target information as string and give a numerical identifier to each)
y = beh_label['labels']

# Identify the resting state
nonrest_task_mask = (y != 'rest')

# Remove the resting state and find names of remaining active labels
categories = y[nonrest_task_mask].unique()
#session = session[nonrest_task_mask]

# Get the labels of the numerical conditions represented by the vector y
unique_conditions, order = np.unique(categories, return_index=True)

# Sort the conditions by the order of appearance
unique_conditions = unique_conditions[np.argsort(order)]

# Extract tags indicating to which acquisition run a tag belongs
session_labels = beh_label['chunks'][nonrest_task_mask]


# In[6]:


# Print basic information on the dataset
print('Functional nifti images are located at: %s' % haxby_ds.func[0])
print('Mask nifti image (3D) is located at:  %s' % haxby_ds.mask)
print('First subject functional nifti images (4D) are at: %s' %func_file)  # 4D data


# In[7]:


# Checkout the confounds of the data
session_target = pd.read_csv(haxby_ds['session_target'][0], sep='\t')

session_target.head()


# ## Preparing the fMRI data (smooth and apply the mask)

# In[8]:


# For decoding, standardizing is important, I am also smoothing the data

nifti_masker = NiftiMasker(mask_img=mask_file, standardize=True, sessions=session,  smoothing_fwhm=4,
                           memory="nilearn_cache", memory_level=1)

X = nifti_masker.fit_transform(func_file)


# Remove the resting state
#X = X[nonrest_task_mask]


# ## Plot Haxby masks

# In[44]:


masker = NiftiMasker(mask_img=mask_vt_file, standardize=True)
fmri_masked = masker.fit_transform(func_file)

# Report depict the computed mask
# masker.generate_report()


# In[10]:


# The variable “fmri_masked” is a numpy array
print(fmri_masked)


# In[11]:


print(fmri_masked.shape)


# ## Converting the Mask to a Matrix

# In[12]:


# load bold image into memory as a nibabel image
func = nib.load(func_file)

# load mask image into memory as a nibabel image
mask = nib.load(mask_file) 

# get the physical data of the mask (3D matrix of voxels)
mask_data = mask.get_data() 

print(func.shape)
print(mask.shape) 
print(len(mask_data[mask_data==1]))


# In[13]:


# Create the masker object 
masker = NiftiMasker(mask_img=mask_file, standardize=True)

# Create a numpy matrix from the BOLD data, using the mask for the transformation
#%memit bold_masked = masker.fit_transform(bold_path)
func_masked = masker.fit_transform(func_file)

# View the dimensions of the matrix. The shape represents the number of time-stamps by the number of voxels in the mask. 
print(func_masked.shape)


# In[14]:


# Viewing the numerical values of the matrix
print(func_masked)


# In[15]:


# Load the labels from a csv into an array using pandas
stimuli = pd.read_csv(labels, sep=' ')


# In[16]:


# View the dimensions of the matrix
print(stimuli.shape)

# Viewing the values of the matrix
print(stimuli)


# In[17]:


targets = stimuli['labels']
print(targets)


# In[18]:


targets_mask = targets.isin(['face', 'cat'])
print(targets_mask)


# In[19]:


func_masked = func_masked[targets_mask]
func_masked.shape


# In[20]:


targets_masked = targets[targets_mask]

print(targets_masked.shape)
print(targets_masked)


# ## Decoding with ANOVA + SVM: face vs house in the Haxby dataset

# In[21]:


# Restrict the analysis to faces and places
condition_mask = beh_label['labels'].isin(['face', 'house'])
conditions_f_h = y[condition_mask]

# Confirm that I now have 2 conditions
print(conditions_f_h.unique())

# Record these as an array of sessions, with fields
# for condition (face or house) and run
session_f_h = beh_label[condition_mask].to_records(index=False)
print(session_f_h.dtype.names)


# In[22]:


# Apply our condition_mask to fMRI data
X_f_h = X[condition_mask]


# In[23]:


#Build the decoder

#Define the prediction function to be used. Here I am using a Support Vector Classification, with a linear kernel
svc = SVC(kernel='linear')

# Define the dimension reduction to be used. (keep 5% of voxels)
feature_selection = SelectPercentile(f_classif, percentile=5)

# I have SVC classifier and our feature selection (SelectPercentile),then plug them together in a *pipeline*:
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])


# In[24]:


# Fit the decoder and predict
anova_svc.fit(X_f_h, conditions_f_h)
y_pred = anova_svc.predict(X_f_h)


# In[25]:


# Obtain prediction scores via cross validation

# Define the cross-validation scheme used for validation, using LeaveOneGroupOut cross-validation.
cv = LeaveOneGroupOut()

# Compute the prediction accuracy for the different folds (i.e. session)
cv_scores = cross_val_score(anova_svc, X_f_h, conditions_f_h, cv=cv, groups=session_f_h)

# Return the corresponding mean prediction accuracy
classification_accuracy = cv_scores.mean()

# Print the results
print("Classification accuracy: %.4f / Chance level: %f" % (classification_accuracy, 1. / len(conditions_f_h.unique())))


# Visualizing the results:

# In[26]:


# Look at the SVC’s discriminating weights
coef = svc.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)
# reverse masking
weight_img = masker.inverse_transform(coef)

# Use the mean image as a background to avoid relying on anatomical data
mean_img = image.mean_img(func_file)

# Create the figure
plot_stat_map(weight_img, mean_img, title='SVM weights')

# Save the results as a Nifti file
#weight_img.to_file('haxby_face_vs_house.nii')


# ## ROI-based decoding analysis
# In this section, I am looking at decoding accuracy for different objects in three different masks: the full ventral stream (mask_vt), the house selective areas (mask_house), and the face-selective areas (mask_face), that have been defined via a standard General Linear Model (GLM) based analysis.

# In[27]:


# extract tags indicating to which acquisition run a tag belongs
session_labels = beh_label["chunks"][nonrest_task_mask]


# In[28]:


# The classifier: a support vector classifier
classifier = SVC(C=1., kernel="linear")

# A classifier to set the chance level
dummy_classifier = DummyClassifier()

# Make a data splitting object for cross validation
cv = LeaveOneGroupOut()

mask_names = ['mask_vt', 'mask_face', 'mask_house']

mask_scores = {}
mask_chance_scores = {}

for mask_name in mask_names:
    print("Working on mask %s" % mask_name)
    
    # Standardizing
    mask_filename = haxby_ds[mask_name][0]
    masker = NiftiMasker(mask_img=mask_filename, standardize=True)
    masked_timecourses = masker.fit_transform(func_file)[nonrest_task_mask]

    mask_scores[mask_name] = {}
    mask_chance_scores[mask_name] = {}

    for category in categories:
        print("Processing %s %s" % (mask_name, category))
        classification_target = (y[nonrest_task_mask] == category)
        mask_scores[mask_name][category] = cross_val_score(
            classifier,
            masked_timecourses,
            classification_target,
            cv=cv,
            groups=session_labels,
            scoring="roc_auc",
        )

        mask_chance_scores[mask_name][category] = cross_val_score(
            dummy_classifier,
            masked_timecourses,
            classification_target,
            cv=cv,
            groups=session_labels,
            scoring="roc_auc",
        )

        print("Scores: %1.2f +- %1.2f" % (
            mask_scores[mask_name][category].mean(),
            mask_scores[mask_name][category].std()))


# ## Different multi-class strategies
# I compare one vs all and one vs one multi-class strategies: the overall cross-validated accuracy and the confusion matrix.

# In[29]:


# Build the decoders, using scikit-learn

svc_ovo = OneVsOneClassifier(Pipeline([
    ('anova', SelectKBest(f_classif, k=500)),
    ('svc', SVC(kernel='linear'))
]))

svc_ova = OneVsRestClassifier(Pipeline([
    ('anova', SelectKBest(f_classif, k=500)),
    ('svc', SVC(kernel='linear'))
]))


# In[30]:


# Remove the "rest" condition
y = y[nonrest_task_mask]
X = X[nonrest_task_mask]

cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=5, verbose=1)

cv_scores_ova = cross_val_score(svc_ova, X, y, cv=5, verbose=1)

print('OvO:', cv_scores_ovo.mean())
print('OvA:', cv_scores_ova.mean())


# In[31]:


# Plot barplots of the prediction scores

plt.figure(figsize=(4, 3))
plt.boxplot([cv_scores_ova, cv_scores_ovo])
plt.xticks([1, 2], ['One vs All', 'One vs One'])
plt.title('Prediction: accuracy score')


# Plot a confusion matrix:

# In[32]:


# I fit on the the first 10 sessions and plot a confusion matrix on the last 2 sessions

svc_ovo.fit(X[session_labels < 10], y[session_labels < 10])
y_pred_ovo = svc_ovo.predict(X[session_labels >= 10])

plot_matrix(confusion_matrix(y_pred_ovo, y[session_labels >= 10]),
            labels=unique_conditions,
            title='Confusion matrix: One vs One', cmap='hot_r')

svc_ova.fit(X[session_labels < 10], y[session_labels < 10])
y_pred_ova = svc_ova.predict(X[session_labels >= 10])

plot_matrix(confusion_matrix(y_pred_ova, y[session_labels >= 10]),
            labels=unique_conditions,
            title='Confusion matrix: One vs All', cmap='hot_r')


# ## Different classifiers for decoding
# In this section I am willing to compare the different classifiers on a visual object recognition decoding task.

# In[33]:


# Standardizing
masker = NiftiMasker(mask_img=mask_vt_file, standardize=True)
masked_timecourses = masker.fit_transform(func_file)[nonrest_task_mask]


# In[34]:


# Support vector classifier
svm = SVC(C=1., kernel="linear")

# The logistic regression
logistic = LogisticRegression(C=1., penalty="l1", solver='liblinear')
logistic_50 = LogisticRegression(C=50., penalty="l1", solver='liblinear')
logistic_l2 = LogisticRegression(C=1., penalty="l2", solver='liblinear')

# Cross-validated versions of these classifiers
# GridSearchCV is slow, but note that it takes an 'n_jobs' parameter that
# can significantly speed up the fitting process on computers with
# multiple cores
svm_cv = GridSearchCV(SVC(C=1., kernel="linear"),
                      param_grid={'C': [.1, 1., 10., 100.]},
                      scoring='f1', n_jobs=1, cv=3, iid=False)

logistic_cv = GridSearchCV(
        LogisticRegression(C=1., penalty="l1", solver='liblinear'),
        param_grid={'C': [.1, 1., 10., 100.]},
        scoring='f1', cv=3, iid=False,
        )
logistic_l2_cv = GridSearchCV(
        LogisticRegression(C=1., penalty="l2", solver='liblinear'),
        param_grid={
            'C': [.1, 1., 10., 100.]
            },
        scoring='f1', cv=3, iid=False,
        )

# The ridge classifier has a specific 'CV' object that can set it's parameters faster than using a GridSearchCV
ridge = RidgeClassifier()
ridge_cv = RidgeClassifierCV()

# A dictionary, to hold all our classifiers
classifiers = {'SVC': svm,
               'SVC cv': svm_cv,
               'log l1': logistic,
               'log l1 50': logistic_50,
               'log l1 cv': logistic_cv,
               'log l2': logistic_l2,
               'log l2 cv': logistic_l2_cv,
               'ridge': ridge,
               'ridge cv': ridge_cv
               }


# Prediction scores:

# In[35]:


# Run time for all these classifiers

# Make a data splitting object for cross validation
cv = LeaveOneGroupOut()

classifiers_scores = {}

for classifier_name, classifier in sorted(classifiers.items()):
    classifiers_scores[classifier_name] = {}
    print(70 * '_')

    for category in categories:
        classification_target = y[nonrest_task_mask].isin([category])
        t0 = time.time()
        classifiers_scores[classifier_name][category] = cross_val_score(
            classifier,
            masked_timecourses,
            classification_target,
            cv=cv,
            groups=session_labels,
            scoring="f1",
        )

        print(
            "%10s: %14s -- scores: %1.2f +- %1.2f, time %.2fs" %
            (
                classifier_name,
                category,
                classifiers_scores[classifier_name][category].mean(),
                classifiers_scores[classifier_name][category].std(),
                time.time() - t0,
            ),
        )


# In[36]:


# Make a rudimentary diagram

plt.figure()

tick_position = np.arange(len(categories))
plt.xticks(tick_position, categories, rotation=45)

for color, classifier_name in zip(
        ['b', 'c', 'm', 'g', 'y', 'k', '.5', 'r', '#ffaaaa'],
        sorted(classifiers)):
    score_means = [classifiers_scores[classifier_name][category].mean()
                   for category in categories]
    plt.bar(tick_position, score_means, label=classifier_name, width=.11, color=color)
    tick_position = tick_position + .09

plt.ylabel('Classification accurancy (f1 score)')
plt.xlabel('Visual stimuli category')
plt.ylim(ymin=0)
plt.legend(loc='lower center', ncol=3)
plt.title('Category-specific classification accuracy for different classifiers')
plt.tight_layout()


# In[37]:


# Plot the face vs house map for the different classifiers

mean_epi_img = image.mean_img(func_file)

# Restrict the decoding to face vs house
condition_mask = y.isin(['face', 'house'])
masked_timecourses = masked_timecourses[
    condition_mask[nonrest_task_mask]]
y_f = (y[condition_mask] == 'face')
# Transform the stimuli to binary values
y_f.astype(np.int)

for classifier_name, classifier in sorted(classifiers.items()):
    classifier.fit(masked_timecourses, y_f)

    if hasattr(classifier, 'coef_'):
        weights = classifier.coef_[0]
    elif hasattr(classifier, 'best_estimator_'):
        weights = classifier.best_estimator_.coef_[0]
    else:
        continue
    weight_img = masker.inverse_transform(weights)
    weight_map = get_data(weight_img)
    threshold = np.max(np.abs(weight_map)) * 1e-3
    plot_stat_map(weight_img, bg_img=mean_epi_img, display_mode='z', cut_coords=[-15],
                  threshold=threshold, title='%s: face vs house' % classifier_name)


# # ML Models

# ### k-Nearest Neighbours

# In[38]:


# Shuffling
X_train, X_test, y_train, y_test=train_test_split(func_masked, targets_masked, test_size=0.1, random_state=42, shuffle=True)


# In[39]:


# Creating and fitting the nearest-neighbor classifier
knn = KNeighborsClassifier() 
knn.fit(X_train, y_train)


# In[40]:


y_pred = knn.predict(X_test)

scores = accuracy_score(y_test, y_pred)
print(scores)


# ### Logistic Regression

# In[41]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

scores = accuracy_score(y_test, y_pred)
print(scores)


# ### Support Vector Machines

# In[42]:


svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

scores = accuracy_score(y_test, y_pred)
print(scores)


# ### Neural Networks

# In[43]:


nn = MLPClassifier(hidden_layer_sizes=(100, 3))
nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)

scores = accuracy_score(y_test, y_pred)
print(scores)


# In[ ]:





# ### Links to the turials:
# 
# Plotting the used stimuli Haxby dataset: #https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_stimuli.html#sphx-glr-auto-examples-02-decoding-plot-haxby-stimuli-py
# 
# Plot Haxby masks: https://nilearn.github.io/auto_examples/01_plotting/plot_haxby_masks.html#sphx-glr-auto-examples-01-plotting-plot-haxby-masks-py
# 
# Decoding with ANOVA + SVM: https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_anova_svm.html#sphx-glr-auto-examples-02-decoding-plot-haxby-anova-svm-py
# 
# ROI-based decoding analysis: https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_full_analysis.html#roi-based-decoding-analysis-in-haxby-et-al-dataset
# 
# Different multi-class strategies: https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_multiclass.html#sphx-glr-auto-examples-02-decoding-plot-haxby-multiclass-py
# 
# Different classifiers for decoding: https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_different_estimators.html#sphx-glr-auto-examples-02-decoding-plot-haxby-different-estimators-py
# 
# Preparing the data: https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html
# 
# k-Nearest Neighbours:
# 
# train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# 
# KNeighborsClassifier(): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# 
# accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 
# LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# Support Vector Machine: 
# https://scikit-learn.org/stable/modules/svm.html 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# 
# Neural Network:
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html 
# MLPClassifier: 
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

# In[ ]:




