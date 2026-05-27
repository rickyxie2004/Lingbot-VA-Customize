# RETURN_PRED_VIDEO=1 PROFILE_ACTION_LATENCY=1 bash evaluation/robotwin/launch_server.sh
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

PROFILE_ACTION_LATENCY=${PROFILE_ACTION_LATENCY:-0}
PROFILE_ACTION_LATENCY_STEPS=${PROFILE_ACTION_LATENCY_STEPS:-0}
RETURN_PRED_VIDEO=${RETURN_PRED_VIDEO:-0}
PROFILE_ARGS=()

if [[ "$PROFILE_ACTION_LATENCY" == "1" ]]; then
    PROFILE_ARGS+=(--profile-action-latency)
fi

if [[ "$PROFILE_ACTION_LATENCY_STEPS" == "1" ]]; then
    PROFILE_ARGS+=(--profile-action-latency-steps)
fi

if [[ "$RETURN_PRED_VIDEO" == "1" ]]; then
    PROFILE_ARGS+=(--return-pred-video)
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root \
    "${PROFILE_ARGS[@]}"


