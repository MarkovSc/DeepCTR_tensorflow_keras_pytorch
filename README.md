# DeepCTR - with tensorflow, keras, pytorch
markov_alg@163.com implement the commom ctr models using tensorflow, keras and pytorch and compare it with python deepctr libaray. The experiment is on the kaggle data.

DeepCTR self:
    Wide & Deep, DeepFM, DCN, XDeepFM

# Experiment:
## File
- python_deep_solution.py: 

    use the python deepctr example usage to run it.
    https://deepctr-doc.readthedocs.io/en/latest/Examples.html
    just install deepctr : pip install deepctr and run it
- kaggle_solution.py:

    kaggle solution by https://www.kaggle.com/snowdog/old-school-nnet

- my_solution.ipynb:

    my solution use the self deepctr

## Data Background
the dataset is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/overview/description
In this competition, you’re challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year.
A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.

## Data Description
in the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.

## Data Example:
id,target,ps_ind_01,ps_ind_02_cat,ps_ind_03,ps_ind_04_cat,ps_ind_05_cat,ps_ind_06_bin,ps_ind_07_bin,ps_ind_08_bin,ps_ind_09_bin,ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin,ps_ind_13_bin,ps_ind_14,ps_ind_15,ps_ind_16_bin,ps_ind_17_bin,ps_ind_18_bin,ps_reg_01,ps_reg_02,ps_reg_03,ps_car_01_cat,ps_car_02_cat,ps_car_03_cat,ps_car_04_cat,ps_car_05_cat,ps_car_06_cat,ps_car_07_cat,ps_car_08_cat,ps_car_09_cat,ps_car_10_cat,ps_car_11_cat,ps_car_11,ps_car_12,ps_car_13,ps_car_14,ps_car_15,ps_calc_01,ps_calc_02,ps_calc_03,ps_calc_04,ps_calc_05,ps_calc_06,ps_calc_07,ps_calc_08,ps_calc_09,ps_calc_10,ps_calc_11,ps_calc_12,ps_calc_13,ps_calc_14,ps_calc_15_bin,ps_calc_16_bin,ps_calc_17_bin,ps_calc_18_bin,ps_calc_19_bin,ps_calc_20_bin
7,0,2,2,5,1,0,0,1,0,0,0,0,0,0,0,11,0,1,0,0.7,0.2,0.7180703307999999,10,1,-1,0,1,4,1,0,0,1,12,2,0.4,0.8836789178,0.3708099244,3.6055512755000003,0.6,0.5,0.2,3,1,10,1,10,1,5,9,1,5,8,0,1,1,0,0,1
9,0,1,1,7,0,0,0,0,1,0,0,0,0,0,0,3,0,0,1,0.8,0.4,0.7660776723,11,1,-1,0,-1,11,1,1,2,1,19,3,0.316227766,0.6188165191,0.3887158345,2.4494897428,0.3,0.1,0.3,2,1,9,5,8,1,7,3,1,1,9,0,1,1,0,1,0

## Test 
the best score in leadboard is 0.29698
submit:
kaggle competitions submit -c porto-seguro-safe-driver-prediction -f submission.csv -m "Message"