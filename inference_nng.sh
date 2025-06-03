prediction_length=41 # 31

exp_dir='./exp'
config='OneForecast'

run_num='20241008-171138'
finetune_dir=''
year=2020
ics_type='default' # options: default, datetime


CUDA_VISIBLE_DEVICES=2 python inference_nng.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --ics_type=${ics_type}



