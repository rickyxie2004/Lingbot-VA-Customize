# PYTHONPATH=$PWD:$PYTHONPATH bash evaluation/libero/launch_client.sh
START=0
END=1
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
PRINT_ACTION_PROFILE=${PRINT_ACTION_PROFILE:-0}
PROFILE_ARGS=()

if [[ "$PRINT_ACTION_PROFILE" == "1" ]]; then
    PROFILE_ARGS+=(--print-profile)
fi

python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port 29056 \
    --host $SERVER_HOST \
    --test-num 2 \
    --task-range $START $END \
    --out-dir outputs/libero \
    "${PROFILE_ARGS[@]}"
