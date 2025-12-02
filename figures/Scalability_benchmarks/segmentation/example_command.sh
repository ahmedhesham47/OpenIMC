#!/bin/bash
# Example command for running segmentation scalability benchmarks
# with the specified nuclear and cytoplasmic channels

python segmentation_scalability.py \
    --images-dir /home/dean/Downloads/OMEandSingleCellMasks/OMEnMasks/ome/ \
    --output-dir ./results \
    --nuclear-channels "Histone_1261726In113Di,Histone_473968La139Di,Histone_phospho_383738Eu153Di,Iridium_10331253Ir191Di,Iridium_10331254Ir193Di" \
    --cyto-channels "Cytoker_651779Pr141Di,Cytoker_3111576Nd143Di,Cytoker_971099Nd144Di,Keratin_346876Sm147Di,CD68_77877Nd146Di,SMA_174864Nd148Di,Vimenti_1921755Sm149Di,c-erbB-_201487Eu151Di,CD3epsi_8001752Sm152Di,Progest_312878Gd158Di,CD44_6967Gd160Di,CD45_71790Dy162Di,CD20_361077Dy164Di,E-Cadhe_1031747Er167Di,panCyto_234832Lu175Di,Cytoker_98922Yb174Di" \
    --num-images 50 100 200 500 \
    --num-workers 1 4 8 16 22 \
    --repeats 3 \
    --cellpose-model cyto3 \
    --channel-format CHW

# Note: The script uses:
# - Mean combination for both nuclear and cytoplasmic channels
# - Channelwise min-max scaling for normalization

