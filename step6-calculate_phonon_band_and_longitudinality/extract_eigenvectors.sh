#!/bin/bash

yaml_file="band.yaml"

qpoints=($(grep -n "q-position" $yaml_file | cut -d: -f1))
total_qpoints=${#qpoints[@]}

for i in "${!qpoints[@]}"; do
    qpoint_line=${qpoints[$i]}
    if [[ $i -lt $(($total_qpoints - 1)) ]]; then
        next_qpoint_line=${qpoints[$((i + 1))]}
        sed -n "$qpoint_line,${next_qpoint_line}p" $yaml_file > "qpoint_$i.dat"
    else
        sed -n "$qpoint_line,\$p" $yaml_file > "qpoint_$i.dat"
    fi

    echo "Data for q_point numer $((i + 1))/$total_qpoints has been saved to qpoint_$i.dat"
done


