FNR == 1 {
    n = FILENAME
    sub(/^.*-/, "", n)
    sub(/[.]out$/, "", n)
    n = int(n)
    all[n] = 1
}

/skipping/ { skipping[n]++ }
/finished/ { finished[n]++ }
/terminated/ { terminated[n]++ }
/Processing/ {
    sub(/:$/, "", $3)
    progress[n] = $3
    sub(/^mem=/, "", $5)
    mem[n] = $5
    sub(/^time=/, "", $7)
    time[n] = $7
}

function sum(a) {
    x = 0
    for (i in a) x += a[i]
    return x
}

END {
    printf " TASK   FINISHED   SKIPPED   TERMINATED       PROGRESS     PEAK MEM   TIME\n"
    fstr = "%5s   %8d   %7d   %10d   %12s   %6s GiB   %s\n"
    for (i in all) {
        printf fstr,
            i,
            finished[i],
            skipping[i],
            terminated[i],
            progress[i],
            mem[i],
            time[i]
    }
    for (i in progress) {
        split(progress[i], pr, "/")
        total_done += pr[1]
        total_all += pr[2]
    }
    for (i in mem) {
        total_mem += mem[i]
    }
    printf fstr,
        "total",
        sum(finished),
        sum(skipping),
        sum(terminated),
        total_done"/"total_all,
        total_mem,
        ""
}
