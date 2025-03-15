# PSAMIL
python version : 3.10

Requirements:
- numpy
- scikit-learn
- torch
- torchvision
- scipy
- PIL 
- visdom
- pandas
- pingouin

## Public Dataset Experiments

### Simple Benchmarks MUSK1,MUSK2,FOX,TIGER,ELEPHANT

1. In the bag-level evaluation, our training and evaluation code and the benchmark datasets are mainly downloaded from different online resources. We recently received a notice that the data input we used during train contain an extra feature dimension (e.g., MUSK1 dataset with `x_train`=167 dimensions = 166 +1(instance label), which is placed in the raw data)).  This specific extra dimension literally contains the instance label(not bag label). **We've already submitted the performance on this input version throughout the review process with all methods re-implemented before we receive the notice, so we explicitly further explain the details in Appendix of paper and here.**

2. A more common implementation of the evaluation code in this version do not include instance label feature. Generally, considering that the mining of these instance label features is still meaningful, the performance rank we presented is similar to the input version before. We also present a 3-rd replication by [TRMIL](https://arxiv.org/abs/2307.14025) here.

**REPLICATED UNIFIED VERSION WITHOUT INSTANCE LABEL:**

A replicated result provided by [TRMIL](https://arxiv.org/abs/2307.14025) is as follows:

![image](https://github.com/user-attachments/assets/de642c9f-7ed3-4035-ade9-31c386778246)


Evaluation:
Run `benchmark.py` to evaluate the performance. 

To evaluate PSMIL under this trimmed data version, unleash the lines below in  `benchmark.py`, also remember to modify the `dim_orig` in `model\PSMIL_benchmark.py`:
```
                        # trim last dim of instance label
                        # ix = ix[:,:-1]
```
After that, you may get a possible performance output under trimmed input version:
```
D:\study\codes\work1\RGMIL-main\venv\Scripts\python.exe D:\study\codes\work1\RGMIL-main\benchmark.py 
D:\study\codes\work1\RGMIL-main\venv\lib\site-packages\torch\optim\adam.py:77: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten\src\ATen/core/TensorBody.h:491.)
  if p.grad is not None:
D:\study\codes\work1\RGMIL-main\venv\lib\site-packages\torch\optim\optimizer.py:459: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten\src\ATen/core/TensorBody.h:491.)
  if p.grad is not None:
overall PSA MUSK1   acc 0.9617777777777777,std:0.06487062147740068
overall PSA MUSK2   acc 0.9638181818181818,std:0.056612062494089635
overall PSA FOX   acc 0.7339999999999999,std:0.13580868897091966
overall PSA TIGER   acc 0.884,std:0.06118823416311341
overall PSA ELEPHANT   acc 0.9179999999999999,std:0.051730068625510245

Process finished with exit code 0
```


### Camelyon16/TCGA-Lung-Cancer
The medical experiments are performed by applying Probability-space Attention(`model\PSMIL.py`) only. It would:

1.Provide explicit inference output for each instance inside, without damage to decent bag-level performance.

2.Provide better visualization and interpretability under multi-instance mode. 

3.The performance of PSMIL may exhibit more fluctuation in certain experiments, meanwhile the likelihood of achieving high performance is greater than that of the original attention models.

#### Introduction
These are two large-scale publicly available medical datasets. The detailed training processes are presented in the directories `logs/CAMELYON16.log` and `logs/TCGA.log`. The CAMELYON16 dataset contains an independent test set that does not rely on random splitting, which is why we provide the independent validation and visualization code `visualize_psmil_c16.py` for the CAMELYON16 data. The expected output is as follows:

```
D:\study\codes\work2\IMIPL\venv\Scripts\python.exe D:\study\codes\work2\IMIPL\testingnewc16.py 
 Testing bag [128/129] bag loss: 0.0009ROC AUC score: 0.9645408163265307
ROC AUC score: 0.9645408163265307
 Testing bag [128/129] bag loss: 0.0156ROC AUC score: 0.9043367346938775
ROC AUC score: 0.9576530612244898
 Testing bag [128/129] bag loss: 0.0000ROC AUC score: 0.9409438775510204
ROC AUC score: 0.9479591836734693
 Testing bag [128/129] bag loss: 0.0000ROC AUC score: 0.9494897959183674
ROC AUC score: 0.9599489795918368
 Testing bag [128/129] bag loss: 0.0014ROC AUC score: 0.9262755102040817
ROC AUC score: 0.9497448979591837
Hamming Loss: 0.07751937984496124
Test Accuracy (Exact Match Ratio): 0.9224806201550387 0.9213537669383296
D:\study\codes\work2\IMIPL\testingnewc16.py:152: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
  mode_result = mode(predictions_stack, axis=0)

Process finished with exit code 0
```

The expected visualization is as follows for test026.jpg:

![image](https://github.com/user-attachments/assets/45a97a48-88ce-45a0-a303-439c793dd595)


PSMIL provides explicit probability outputs for each instance(slide patch), making the visualization very clear. We also provided the test slide visualization results of DSMIL in `test-c16\output`, which corresponds to the files end with "_ds". These results are generated by attention score, so you may see many unsure regions compared to PSMIL. 

You could also validate the weights on TCGA dataset(also stored in `weights/20241126`, with the weights file name corresponding to `logs/TCGA.log`) by customize your own code. But note that there is no fixed test set for TCGA so you may get slightly different performance.

#### How to

Visualization of Camelyon16:

1. Download the file [test-c16](https://pan.baidu.com/s/1zQlGNDUPnEyoUv-WnQP70A?pwd=cdgf )(extract code: cdgf) 
and unzip the contents to directory `test-c16` to provide the independent test data needed.
2. You could also download the raw test set slides of C16 dataset and proceed by your own to produce the patches you need and put corresponding files to `test-c16`. If you choose to do so, go to [DSMIL](https://github.com/binli123/dsmil-wsi) and follow the detailed instructions there.

3. run `visualize_psmil_c16.py` to provide metrics and visualization.

Train:

1. Download the files [TCGA-Dataset](https://uwmadison.box.com/shared/static/tze4yqclajbdzjwxyb8b1umfwk9vcdwq.zip) and [Camelyon16_Dataset](https://uwmadison.box.com/shared/static/l9ou15iwup73ivdjq0bc61wcg5ae8dwe.zip) to directory `datasets`, and unzip both.

2. run `traincancer.py`


The training/evaluation codes and data are mostly modified from previous work [DSMIL](https://github.com/binli123/dsmil-wsi). We would make more instructions here to present our process more clearly:

1. We used the SimCLR 20x features provided by [DSMIL]([https://github.com/binli123/dsmil-wsi]) in training set.
2. We used `model/PSMIL.py` to implement the experiments, which only involves probability space attention.
3. We modified the original code from DSMIL when split the test bags for Camelyon16, because it is different with the mainstream evaluation way. We choose the whole test set as the `reserved_testing_bags` variable in `traincancer.py`. This is different from the original "Camelyon16 with a 5-fold cross-validation and a standalone test set" introduced by [DSMIL](https://github.com/binli123/dsmil-wsi), which choose a random inner split as test set.
4. In `visualize_psmil_c16.py`, we loaded the initial embedder weights provided by [DSMIL]([https://github.com/binli123/dsmil-wsi]) to produce the test slide SimCLR 20x features. To simplify the codes, we directly use the SimCLR features generated in the middle of DSMIL as the input to our model. Details see the code.




## Validation on Simulated Dataset,Instance level
logs: `logs\simulated_cifar10\probalign-fsa.log` corresponds to feature space attention with prob align term; `logs\simulated_cifar10\probalign-psa.log` corresponds to probability space attention with prob align term; suffix `RL` means unfreezing the encoder.

The model was proposed by optimizing our simulated data performance, so here is our advice and validation through the process:

1. It is **POSSIBLE** for your deep attention-based MIL model to degrade to random guessing. If your model faces degradation issue when unfreeze the feature encoder, consider to augment your input and align to achieve stable performance (top acc 81% in `logs\simulated_cifar10\probaligned-fsa.log`).

2. There **ARE** cases when classic attention-based MIL model trained successfully (top acc 70% in `logs\simulated_cifar10\Not-degraded-fsa-RL.log`, top acc 69% in `logs\simulated_cifar10\fixed_feature_fsa.log`);  also applying Probability-space Attention **ONLY** may be not effective enough to fully avoid degradation.
   
3. When using probability-space attention together with prob align, fix the encoder parameter in the first epoch to provide better initial estimation. **A careful tune of hyperparameter in this version would bring the best performance among all models we tested on instance level, also we found NO degradation issue under this setting.(top acc 88% in `logs\simulated_cifar10\probaligned-psa.log`)**.
   
4. The weights version of psa we upload would provide around 87% accuracy(5% gap to full supervision, top acc 92% in `logs\simulated_cifar10\Res18-Fullsup.log`), we are not able to provide the exact weights in the `logs\simulated_cifar10\probaligned-psa.log` because we have **LOST** original weights.  

Run  `visualize_psmil_cifar.py` to test the performance of `model/PSAMIL`. You can also switch the supported param by your own to validate the other version.

Weights on [Netdisk](https://pan.baidu.com/s/1tJAx_sqN7rSDopzGmiv0Iw?pwd=wu5y)(Extract code:wu5y).


