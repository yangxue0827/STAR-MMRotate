set -x

partition='VC2'
TYPE='reserved'
# partition='INTERN3'
# TYPE='spot'
JOB_NAME='gliding_vertex_r50_fpn_1x_rsg_le90'
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=12
export MASTER_PORT=29511

SRUN_ARGS="--jobid=3211252 --nodes=1"
CONFIG="./configs/gliding_vertex/gliding_vertex_r50_fpn_1x_rsg_le90.py"
CHECKPOINT="./work_dirs/gliding_vertex_r50_fpn_1x_rsg_le90/latest.pth"

EXTRA_ARGS="--eval mAP"

http_proxy=http://liqingyun:yun_990608@10.1.8.50:33128/
https_proxy=http://liqingyun:yun_990608@10.1.8.50:33128/
HTTP_PROXY=http://liqingyun:yun_990608@10.1.8.50:33128/
HTTPS_PROXY=http://liqingyun:yun_990608@10.1.8.50:33128/

WANDB_API_KEY="b1e5fa15e4cbf702cfeb8ff3bd58a522c908cebc"
# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

srun -p $partition --job-name=${JOB_NAME} ${SRUN_ARGS} \
  --gres=gpu:${GPUS_PER_NODE} --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} -n${GPUS} \
  --quotatype=${TYPE} --kill-on-bad-exit=1 \
  python ./tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${EXTRA_ARGS}
