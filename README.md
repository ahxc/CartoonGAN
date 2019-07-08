# CartoonGAN
Simple Tensorflow implementation of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.html) (CVPR 2018)

## Environment
python 3.6.3

tensorflow 1.10

windows 10

## Download vgg19
[vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

## Training
To get better results, the training time is much longer than cycleGAN

```
python train.py --dataset persoon2cartoon --img_size 256 --...
```

Different from CycleGAN, CartoonGAN added a fuzzy dataset, I provides a simple script(smoother.py) for generating fuzzy dataset, 

```
python smoother.py --dataset persoon2cartoon --img_size 256 --kernerl_size 5
```

Your data directory structure should look like this

```
├── dataset
   └── YOUR_DATASET_NAME(persoon2cartoon)
       ├── trainA (input)
           ├── yyy.png
           └── ...
       ├── trainB (output)
           ├── www.png
           └── ...
       ├── trainB_smooth (After you run the smoother.py, it will be created automatically)
           ├── www.png
           └── ...
```

## Result preview

<p align="center">
  <img src="/Related images/step-17100.png">
  <img src="/Related images/step-17200.png">
  <img src="/Related images/step-17300.png">
  <img src="/Related images/step-17400.png">
</p>

## References

code: [github-CartoonGAN](https://github.com/taki0112/CartoonGAN-Tensorflow)

blog: [zhihu-CartoonGAN](https://zhuanlan.zhihu.com/p/40725950)

paper: [CVF-CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.html)
