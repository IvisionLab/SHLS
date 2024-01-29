# SHLS: Superfeatures learned from still images for self-supervised VOS

BMVC 2023 paper and materials: http://proceedings.bmvc2023.org/523/

Concept:
![latent_space](https://github.com/marcelo-mendonca/SHLS/assets/38759879/43dcc424-1cfb-4d7b-915e-385b5e569be4)

Superfeatures:
![superfeature](https://github.com/marcelo-mendonca/SHLS/assets/38759879/6ff0e924-801a-4951-aee4-28ed5d6fc7da)


Framework:
![complete_model](https://github.com/marcelo-mendonca/SHLS/assets/38759879/b79ac068-2ad9-475e-bf12-fad186fc10d1)


Requirements:
- pytorch
- opencv
- tensorboardx
- [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- faiss

Weights [here](https://drive.google.com/file/d/1dgdLyKL2EmsMWH-IB4LbCc-nxWZliPuB/view?usp=sharing).

Folder structure:
- root
- databases
   - MSRA
      - MSRA10K_Imgs_GT
         - train.txt 
         - val.txt
         - Imgs
         - saliency
            
    - DAVIS2017
        - Annotations
        - ImageSets
        - JPEGImages

Citation:
   - MENDONÇA, M.; FONTINELE, J.; OLIVEIRA, L. SHLS: Superfeatures learned from still images for self-supervised VOS. In: British Machine Vision Conference (BMVC), 2023. In: http://proceedings.bmvc2023.org/523/
