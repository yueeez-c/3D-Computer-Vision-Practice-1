# CS6501 3D Computer Vision — Exercise 1 (code)

Companion code for the Exercise 1 handout. This is a practice exercise: nothing to hand in.

## Requirements

`numpy` and `matplotlib`. Nothing else, and no downloads — all data is included.

```
pip install numpy matplotlib
```

Everything also runs as-is on Google Colab.

## Layout

```
code/exercise1.ipynb      Tasks A, C, D and the two demos
code/make_data.py         regenerates data/*.npz (already included; only needed if you delete them)
code/p2-demosaic/         Task B — run from the command line, not the notebook
data/demosaic/            ten test images for Task B
data/*.npz                synthetic scenes for Tasks C, D and the demos
```

## Task B — demosaicing

Edit `code/p2-demosaic/demosaicImage.py`: `demosaicBaseline` is given, and
`demosaicNN` / `demosaicLinear` / `demosaicAdagrad` are yours to write (`adagrad` is optional).
Then, **from inside `code/p2-demosaic/`**:

```
cd code/p2-demosaic
python evalDemosaicing.py
```

It mosaics each of the ten images, runs your methods, and prints an error table. Reconstructed
images and error maps are written to `output/demosaic/`. Set `display = True` near the top of
`evalDemosaicing.py` to pop up the error map for each image as it goes.

## Tasks A, C, D and the demos

```
jupyter notebook code/exercise1.ipynb
```

Every place you must write code is marked `# TODO`. Each task ends with a **self-check** cell that
tests your implementation against a hard numerical threshold — run it and make sure it prints
`PASS` before moving on. If a self-check fails, the problem is in your code, not in the checker.

The two demos at the end are already written. Just run them and look at the output.
