# GenLearn
This repo contains the code and notebooks to reproduce the results of the  following papers: https://arxiv.org/abs/2406.09028 and https://chemrxiv.org/engage/chemrxiv/article-details/672cd8425a82cea2fa640adc
The notebooks used to reproduce the results of https://arxiv.org/abs/2406.09028 are in the NeurIPS folder, however, the code for the training is not optimized.

On the other hand, notebooks, with (some) explanation can be found in the alanine_dipeptide and alanine_tetrapeptide folders. They rely on a modified fork of mlcolvar. To use it, 
```console
hello@myworkstation:~$git clone https://github.com/DevergneTimothee/mlcolvar
hello@myworkstation:~$cd mlcolvar
hello@myworkstation:~$pip install -e .
```

The COLVAR files can be found in the data folder, while the input files can be found in the inputs folders of alanine_tetrapeptide and alanine_dipeptide

