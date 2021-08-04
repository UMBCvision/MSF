# MSF
Official code for "Mean Shift for Self-Supervised Learning" https://arxiv.org/abs/2105.07269
<p align="center">
  <img src="https://user-images.githubusercontent.com/62820830/112181641-fd0fdb80-8bd2-11eb-8444-8e0b0547e622.jpg" width="95%">
</p>

```
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
| ------------------ | --------- |----------------------------------| ----------------- | ------- | ----------------- |
| MSF(Resnet50) |               72.4%              |        62.5%        | 65.7% | [Pretrained Resnet50](https://drive.google.com/file/d/105TZ5IKqS9b5d_j2oP2wX0PkuGZ9rVRl/view?usp=sharing) |






# License

This project is under the MIT license.
