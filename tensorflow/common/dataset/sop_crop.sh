#!/usr/bin/bash
DATA_PATH=/project_scratch/diva/2017/data/Stanford_Online_Products
OUT_PATH=/project_scratch/diva/2017/data/SOP/Cropped

JOBS=16
set -x

SUB=bicycle_final 
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=cabinet_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=chair_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=coffee_maker_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=fan_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=kettle_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=lamp_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=mug_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=sofa_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=stapler_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=table_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 

SUB=toaster_final
python -u sop_pre.py  $DATA_PATH/$SUB -o $OUT_PATH/$SUB -j $JOBS 
