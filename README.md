# PSAMIL

## Public Dataset Experiments

### Camelyon16/TCGA-Lung-Cancer

These are two large-scale publicly available medical datasets. The training logs are presented in the directories `logs/CAMELYON16.log` and `logs/TCGA.log` for your validation. The CAMELYON16 dataset contains an independent test set that does not rely on random splitting, which is why we provide the independent validation and visualization code `visualize_psmil_c16.py` for the CAMELYON16 data. The expected output is as follows:

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
![image](https://github.com/user-attachments/assets/a2ee439f-efd4-4585-aaca-061a3bfe6d58)

PSMIL provides probability outputs for each instance(slide patch), making the visualization very clear.

You could also validate the weights on TCGA dataset(also stored in `weights/20241126`, with the file name corresponding to `logs/TCGA.log`) by customize your own code. But note that there is no fixed test set for TCGA so you may get slightly different performance.



You can put MNIST raw dataset under directory "datasets" first.

```python make_ds.py```

It may cost several hours to produce MMNIST.

Run following command to inspect the accuracy float:

```python -m visdom.server```

Run following command to test different aggregators on MMNIST of different modes：

```python main.py --pooling rgp --bagsize 64```

When bagsize set to 1, it corresponds to fully supervised. Choose your options as supported.
