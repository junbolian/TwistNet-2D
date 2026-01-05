\# Dataset Preparation Guide



This guide explains how to download and organize the five benchmark datasets for TwistNet-2D experiments.



\## Overview



| Dataset | Classes | Images | Task | Download Size |

|---------|---------|--------|------|---------------|

| DTD | 47 | 5,640 | Texture Recognition | ~600 MB |

| FMD | 10 | 1,000 | Material Recognition | ~250 MB |

| KTH-TIPS2 | 11 | 4,752 | Material Recognition | ~1.8 GB |

| CUB-200-2011 | 200 | 11,788 | Fine-grained Recognition | ~1.2 GB |

| Flowers-102 | 102 | 8,189 | Fine-grained Recognition | ~330 MB |



\## Directory Structure



After setup, your `data/` folder should look like:

```

data/

├── dtd/

│   └── images/

│       ├── banded/

│       ├── blotchy/

│       └── ...

├── fmd/

│   └── image/

│       ├── fabric/

│       ├── foliage/

│       └── ...

├── kth\_tips2/

│   ├── aluminium\_foil/

│   ├── brown\_bread/

│   └── ...

├── cub200/

│   └── images/

│       ├── 001.Black\_footed\_Albatross/

│       ├── 002.Laysan\_Albatross/

│       └── ...

└── flowers102/

&nbsp;   ├── jpg/

&nbsp;   │   ├── image\_00001.jpg

&nbsp;   │   └── ...

&nbsp;   ├── imagelabels.mat

&nbsp;   └── setid.mat

```



\## Download Instructions



\### 1. DTD (Describable Textures Dataset)

```bash

\# Download

wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz



\# Extract

tar -xzf dtd-r1.0.1.tar.gz -C data/



\# Rename (if needed)

mv data/dtd data/dtd

```



\*\*Alternative:\*\* \[Direct Download Link](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)



\### 2. FMD (Flickr Material Database)

```bash

\# Download from official website

\# https://people.csail.mit.edu/celiu/CVPR2010/FMD/



\# Extract to data/fmd/

```



\*\*Download Link:\*\* \[FMD Dataset](https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip)



\### 3. KTH-TIPS2

```bash

\# Download

wget https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b.tar



\# Extract

tar -xf kth-tips2-b.tar -C data/



\# Rename

mv data/KTH-TIPS2-b data/kth\_tips2

```



\*\*Download Link:\*\* \[KTH-TIPS2-b](https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b.tar)



\### 4. CUB-200-2011 (Caltech-UCSD Birds)

```bash

\# Download

wget https://data.caltech.edu/records/65de6-vp158/files/CUB\_200\_2011.tgz



\# Extract

tar -xzf CUB\_200\_2011.tgz -C data/



\# Rename

mv data/CUB\_200\_2011 data/cub200

```



\*\*Download Link:\*\* \[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158/files/CUB\_200\_2011.tgz)



\### 5. Flowers-102 (Oxford Flowers)

```bash

\# Download images

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz



\# Download labels

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat



\# Download splits

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat



\# Extract

tar -xzf 102flowers.tgz -C data/flowers102/

mv data/flowers102/jpg data/flowers102/

mv imagelabels.mat data/flowers102/

mv setid.mat data/flowers102/

```



\*\*Download Links:\*\*

\- \[Images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)

\- \[Labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)

\- \[Splits](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat)



\## Windows Users



If you're on Windows without `wget`, you can:



1\. \*\*Use PowerShell:\*\*

```powershell

Invoke-WebRequest -Uri "URL" -OutFile "filename"

```



2\. \*\*Use browser:\*\* Click the download links directly



3\. \*\*Use 7-Zip:\*\* Extract `.tar.gz` and `.tgz` files



\## Verification



Run this command to verify your setup:

```bash

python -c "

import os

datasets = \['dtd', 'fmd', 'kth\_tips2', 'cub200', 'flowers102']

for d in datasets:

&nbsp;   path = f'data/{d}'

&nbsp;   if os.path.exists(path):

&nbsp;       n\_files = sum(len(files) for \_, \_, files in os.walk(path))

&nbsp;       print(f'✓ {d}: {n\_files} files')

&nbsp;   else:

&nbsp;       print(f'✗ {d}: NOT FOUND')

"

```



Expected output:

```

✓ dtd: ~5700 files

✓ fmd: ~1000 files

✓ kth\_tips2: ~4800 files

✓ cub200: ~12000 files

✓ flowers102: ~8200 files

```



\## Quick Start After Setup

```bash

\# Test with DTD dataset

python run\_all.py --data\_dir data/dtd --dataset dtd --models twistnet18 --folds 1 --seeds 42 --epochs 10 --run\_dir runs/test

```



\## Troubleshooting



\### "Dataset not found" error

\- Check that the folder names match exactly: `dtd`, `fmd`, `kth\_tips2`, `cub200`, `flowers102`

\- Ensure images are not nested in extra folders



\### "scipy not found" for Flowers-102

```bash

pip install scipy

```



\### Large file extraction issues

\- Use 7-Zip for better compatibility on Windows

\- Ensure you have enough disk space (~5 GB total)



\## Citation



If you use these datasets, please cite the original papers:

```bibtex

@inproceedings{cimpoi2014dtd,

&nbsp; title={Describing textures in the wild},

&nbsp; author={Cimpoi, M. and Maji, S. and Kokkinos, I. and Mohamed, S. and Vedaldi, A.},

&nbsp; booktitle={CVPR},

&nbsp; year={2014}

}



@inproceedings{sharan2009fmd,

&nbsp; title={Material perception: What can you see in a brief glance?},

&nbsp; author={Sharan, L. and Rosenholtz, R. and Adelson, E.},

&nbsp; booktitle={Journal of Vision},

&nbsp; year={2009}

}



@article{caputo2005kthtips,

&nbsp; title={Class-specific material categorisation},

&nbsp; author={Caputo, B. and Hayman, E. and Mallikarjuna, P.},

&nbsp; journal={ICCV},

&nbsp; year={2005}

}



@techreport{wah2011cub200,

&nbsp; title={The Caltech-UCSD Birds-200-2011 Dataset},

&nbsp; author={Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},

&nbsp; institution={California Institute of Technology},

&nbsp; year={2011}

}



@inproceedings{nilsback2008flowers102,

&nbsp; title={Automated flower classification over a large number of classes},

&nbsp; author={Nilsback, M-E. and Zisserman, A.},

&nbsp; booktitle={ICVGIP},

&nbsp; year={2008}

}

```

