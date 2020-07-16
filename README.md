# CʀꜰPᴀʀ

[![Travis](https://img.shields.io/travis/yzhangcs/crfpar.svg)](https://travis-ci.org/yzhangcs/crfpar)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/crfpar.svg)](https://github.com/yzhangcs/crfpar/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yzhangcs/crfpar.svg)](https://github.com/yzhangcs/crfpar/stargazers)		
[![GitHub forks](https://img.shields.io/github/forks/yzhangcs/crfpar.svg)](https://github.com/yzhangcs/crfpar/network/members)

Source code for ACL'20 paper ["Efficient Second-Order TreeCRF for Neural Dependency Parsing"](https://www.aclweb.org/anthology/2020.acl-main.302/) and IJCAI'20 paper ["Fast and Accurate Neural CRF Constituency Parsing"](https://www.ijcai.org/Proceedings/2020/560).

The code of ACL'20 paper (Cʀꜰ2o is not ported yet) and IJCAI'20 paper is available at the [`crf-dependency`](https://github.com/yzhangcs/crfpar/tree/crf-dependency) branch and [`crf-dependency`](https://github.com/yzhangcs/crfpar/tree/crf-constituency) branch respectively.

Currently I'm working to release a python package named `supar`, including pretrained models for my papers.
The code is unstable and not imported to this repo yet.
If you would like to try them out in advance, please refer to my another repository [parser](https://github.com/yzhangcs/parser/tree/release).

## Citation

If you are interested in our researches, please cite:
```
@inproceedings{zhang-etal-2020-efficient,
  title     = {Efficient Second-Order {T}ree{CRF} for Neural Dependency Parsing},
  author    = {Zhang, Yu and Li, Zhenghua and Zhang Min},
  booktitle = {Proceedings of ACL},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.acl-main.302},
  pages     = {3295--3305}
}

@inproceedings{zhang-etal-2020-fast,
  title     = {Fast and Accurate Neural {CRF} Constituency Parsing},
  author    = {Zhang, Yu and Zhou, Houquan and Li, Zhenghua},
  booktitle = {Proceedings of IJCAI},
  year      = {2020},
  doi       = {10.24963/ijcai.2020/560},
  url       = {https://doi.org/10.24963/ijcai.2020/560},
  pages     = {4046--4053}
}
```

Please feel free to email [me](mailto:yzhang.cs@outlook.com) if you have any issues.
