# MSF
Official code for <a href="https://www.csee.umbc.edu/~hpirsiav/papers/MSF_iccv21.pdf"> "Mean Shift for Self-Supervised Learning"</a> accepted as an oral presentation in ICCV 2021. 
<!-- https://arxiv.org/abs/2105.07269 -->
<p align="center">
  <img src="https://user-images.githubusercontent.com/62820830/112181641-fd0fdb80-8bd2-11eb-8444-8e0b0547e622.jpg" width="95%">
</p>

```
@InProceedings{Koohpayegani_2021_ICCV,
    author    = {Koohpayegani, Soroush Abbasi and Tejankar, Ajinkya and Pirsiavash, Hamed},
    title     = {Mean Shift for Self-Supervised Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10326-10335}
}

@misc{koohpayegani2021mean,
      title={Mean Shift for Self-Supervised Learning}, 
      author={Soroush Abbasi Koohpayegani and Ajinkya Tejankar and Hamed Pirsiavash},
      year={2021},
      eprint={2105.07269},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


# Requirements

- Python >= 3.7.6
- PyTorch >= 1.4
- torchvision >= 0.5.0
- faiss-gpu >= 1.6.1

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). We used Python 3.7 for our experiments.


- Install PyTorch ([pytorch.org](http://pytorch.org))


To run NN and Cluster Alignment, you require to install FAISS. 

FAISS: 
- Install FAISS ([https://github.com/facebookresearch/faiss/blob/master/INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md))


# Training



We train on 4 RTX6000 GPUs with 24GB of memory. But one can run our model with 4 RTX 2080Ti GPUs with 11GB of memory as well(with 128K memory bank). 200 Epochs of training with ResNet50 backbone will take approximately 140 hours to train. 



Following command can be used to train the MSF 

```
python train_msf.py \
  --cos \
  --weak_strong \
  --learning_rate 0.05 \
  --epochs 200 \
  --arch resnet50 \
  --topk 10 \
  --momentum 0.99 \
  --mem_bank_size 128000 \
  --checkpoint_path <CHECKPOINT PATH> \
  <DATASET PATH>
```

# Pretrained Models

| Model         | Top-1 Linear Classifier Accuracy | Top-1 Nearest Neighbor Accuracy | Top-1 KNN Accuracy | Link |
| ------------------ | ------- | ------- | ------- | ----------------- |
| MSF(Resnet50) |               72.4%              |        62.5%        | 65.7% | [Pretrained Resnet50](https://drive.google.com/file/d/105TZ5IKqS9b5d_j2oP2wX0PkuGZ9rVRl/view?usp=sharing) |






# License

This project is under the MIT license.
