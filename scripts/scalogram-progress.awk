/^[0-9]+\/[0-9]+$/ {
    split($0, a, "/")
    file_n = a[1]
    file_total = a[2]
}

FNR == 1 {
    total_n += file_n
    all_total += file_total
}

END {
    total_n += file_n
    all_total += file_total
    print total_n"/"all_total
}
