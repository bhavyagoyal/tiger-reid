#!/usr/bin/bash
DATA_PATH=/project_scratch/diva/2017/data/Stanford_Online_Products
OUT_PATH=/project_scratch/diva/2017/data/SOP/Cropped_256

JOBS=16
set -x

SUB=bicycle_final 
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=cabinet_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=chair_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=coffee_maker_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=fan_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=kettle_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=lamp_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=mug_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=sofa_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=stapler_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=table_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256

SUB=toaster_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS -f 256
