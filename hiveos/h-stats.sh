#!/usr/bin/env bash

API_URL="http://localhost:42069/stats"
miner_stat=$(curl -s $API_URL)

gpu_stats=$(gpu-stats)

bus_ids_data=$(echo "$gpu_stats" | jq -r '.busids[]')
temps_data=($(echo "$gpu_stats" | jq -r '.temp[]'))
fans_data=($(echo "$gpu_stats" | jq -r '.fan[]'))

bus_ids=()
for bus_id in $bus_ids_data; do
    hex_bus_id=$(echo "$bus_id" | cut -d':' -f1)
    decimal_bus_id=$((16#$hex_bus_id))
    bus_ids+=("$decimal_bus_id")
done

uptime=$(jq -r '.uptime' <<<"$miner_stat")
total_hashrate=$(jq -r '.totalHashrate' <<<"$miner_stat")
accepted_blocks=$(jq -r '.acceptedBlocks' <<<"$miner_stat")
rejected_blocks=$(jq -r '.rejectedBlocks' <<<"$miner_stat")
gpus=$(jq -c '.gpus[]' <<<"$miner_stat")

stats_hashrates=()
stats_temps=()
stats_fans=()
bus_numbers=()

while IFS= read -r gpu; do
    found=false
    hashrate=$(jq -r '.hashrate' <<<"$gpu")
    device_bus=$(jq -r '.busId' <<<"$gpu")
    stats_hashrates+=("$hashrate")
    for i in "${!bus_ids[@]}"; do
        if [ "$device_bus" -eq "${bus_ids[$i]}" ]; then
            stats_temps+=("${temps_data[$i]}")
            echo "temps_data[$i]=${temps_data[$i]}"
            stats_fans+=("${fans_data[$i]}")
            found=true
            break
        fi
    done
    bus_numbers+=("$device_bus")
    if [ "$found" != true ]; then
        stats_temps+=(0)
        stats_fans+=(0)
    fi
done <<<"$gpus"

khs=$(echo "$total_hashrate" | jq -r '. / 1000')
stats=$(jq -nc \
    --argjson hs "$(echo "${stats_hashrates[@]}" | tr ' ' '\n' | jq -cs '.')" \
    --arg hs_units "hs" \
    --argjson temp "$(echo "${stats_temps[@]}" | tr ' ' '\n' | jq -cs '.')" \
    --argjson fan "$(echo "${stats_fans[@]}" | tr ' ' '\n' | jq -cs '.')" \
    --arg uptime "$uptime" \
    --argjson ar "[$accepted_blocks, $rejected_blocks]" \
    --argjson bus_numbers "$(echo "${bus_numbers[@]}" | tr ' ' '\n' | jq -cs '.')" \
    '{"hs":$hs, "hs_units":$hs_units, "temp":$temp, "fan":$fan, "uptime":$uptime, "ver":"levykrak & Woody", "ar":$ar, "algo":"argon2id", "bus_numbers":$bus_numbers}')
    
# echo "khs=$khs"
# echo "stats=$stats"
