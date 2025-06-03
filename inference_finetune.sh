
prediction_length=41

exp_dir='./exp'
config='OneForecast'
run_num='20241008-171138'
finetune_dir='60_steps_finetune'
ics_type='default'

CUDA_VISIBLE_DEVICES=0 python inference.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --ics_type=${ics_type}



