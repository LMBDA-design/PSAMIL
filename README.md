# PSAMIL

## Public Dataset Experiments

### Camelyon16/TCGA-Lung-Cancer



You can put MNIST raw dataset under directory "datasets" first.

```python make_ds.py```

It may cost several hours to produce MMNIST.

Run following command to inspect the accuracy float:

```python -m visdom.server```

Run following command to test different aggregators on MMNIST of different modes：

```python main.py --pooling rgp --bagsize 64```

When bagsize set to 1, it corresponds to fully supervised. Choose your options as supported.
