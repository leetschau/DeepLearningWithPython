# Intro

This folder is for chapter 5.

I didn't use Python scripts in the origin codes (which is too verbose for this task).

The equivalent shell script is as follows (the training picture files is downloaded
and extracted to folder *~/Downloads/dogs-vs-cats*):
```
mkdir -p train/{cats,dogs} validation/{cats,dogs} test/{cats,dogs}
ls ~/Downloads/dogs-vs-cats/train/cat*|head -1000|xargs -I % cp % train/cats
ls ~/Downloads/dogs-vs-cats/train/cat*|head -1500|tail -500|xargs -I % cp % validation/cats
ls ~/Downloads/dogs-vs-cats/train/cat*|head -2000|tail -500|xargs -I % cp % test/cats
ls ~/Downloads/dogs-vs-cats/train/dog*|head -1000|xargs -I % cp % train/dogs
ls ~/Downloads/dogs-vs-cats/train/dog*|head -1500|tail -500|xargs -I % cp % validation/dogs
ls ~/Downloads/dogs-vs-cats/train/dog*|head -2000|tail -500|xargs -I % cp % test/dogs
```

Then run dogs_cats_cnn.py.

Each epoch takes 55 ~ 70s.
So the first training takes half an hour (30 epochs), the second takes 100 minutes (100 epochs)
on my machine (CPU: Intel i7-6700 3.40GHz, 8 cores, Memory: 32GB).
