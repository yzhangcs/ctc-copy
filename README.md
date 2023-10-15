<div align="center">

# Non-autoregressive Text Editing with Copy-aware Latent Alignments

<div>
  <a href='https://yzhang.site/' target='_blank'><b>Yu Zhang</b></a><sup>1*</sup>&emsp;
  <a href='https://hillzhang1999.github.io/' target='_blank'>Yue Zhang</a><sup>1*</sup>&emsp;
  <a href='https://nealcly.github.io/' target='_blank'>Leyang Cui</b></a><sup>2</sup>&emsp;
  <a href='https://web.suda.edu.cn/ghfu/' target='_blank'>Guohong Fu</a><sup>1</sup>&emsp;
</div>
<div><sup>1</sup>Soochow University, Suzhou, China</div>
<div><sup>2</sup>Tencent AI Lab</div>

<div>
<h4>

[![conf](https://img.shields.io/badge/EMNLP%202023-orange?style=flat-square)](https://yzhang.site/assets/pubs/emnlp/2023/ctc.pdf)
[![arxiv](https://img.shields.io/badge/arXiv-2310.07821-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.07821)
[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F116277fd27c97d50bba2d8023d3c590c1ea8187b%3Ffields%3DcitationCount&style=flat-square)](https://www.semanticscholar.org/paper/Non-autoregressive-Text-Editing-with-Copy-aware-Zhang-Zhang/116277fd27c97d50bba2d8023d3c590c1ea8187b)
![python](https://img.shields.io/badge/python-%3E%3D%203.7-pybadges.svg?logo=python&style=flat-square)

</h4>
</div>

<img width="400" alt="image" src="https://github.com/yzhangcs/ctc-copy/assets/18402347/eb3ff302-aa3a-47b5-983f-2d7b6b4f4370">

</div>

## Citation

If you are interested in our work, please cite
```bib
@inproceedings{zhang-etal-2023-ctc,
  title     = {Non-autoregressive Text Editing with Copy-aware Latent Alignments},
  author    = {Zhang, Yu  and
               Zhang, Yue  and
               Cui, Leyang  and
               Fu, Guohong},
  booktitle = {Proceedings of EMNLP},
  year      = {2023},
  address   = {Singapore}
}
```

## Setup

The following packages should be installed:
* [`PyTorch`](https://github.com/pytorch/pytorch): >= 2.0
* [`Transformers`](https://github.com/huggingface/transformers)
* [`Errant`](https://github.com/chrisjbryant/errant)

Clone this repo recursively:
```sh
git clone https://github.com/yzhangcs/ctc-copy.git --recursive
```

You can follow this [repo](https://github.com/HillZhang1999/SynGEC) to obtain the 3-stage train/dev/test data for training a English GEC model.
The multilingual datasets are available [here](https://github.com/google-research-datasets/clang8).

Before running, you are required to preprocess each sentence pair into the format of `SRC:\t[src]\nTGT:\t[tgt]\n`, where `src` and `tgt` are the source and target sentences, respectively. Each sentence pair is separated by a blank line.
See [`data/clang8.toy`](data/clang8.toy) for examples.

## Run

Try the following command to train a 3-stage English model,
```sh
bash train.sh
```
To make predictions & evaluations:
```sh
bash pred.sh
```

## Contact

If you have any questions, please feel free to [email](mailto:yzhang.cs@outlook.com) me.
