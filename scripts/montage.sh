#!/bin/sh

for dir in ~/abmurraylab/shoreshop3/analysis/wavelets/*/figures/multi-profile/*; do
    for subdir in "$dir"/*; do
        name="${subdir##*/}"
        n=$(ls "$subdir" | wc -l)
        r=$(awk -v x=$n 'BEGIN{print int(sqrt(x))+1;quit}')
        c=$((n/r+1))
        f="$dir/${subdir##*/}-tiled.png"
        echo "$f" $n $r $c
        [ -e "$f" ] || montage "$subdir"/* -tile ${r}x$c -geometry +1+1 "$f"
    done
done