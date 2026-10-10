#!/bin/bash
# Resume a large download using several parallel byte-range connections, then join and verify MD5.
# Usage: parallel_download.sh URL OUTFILE TOTAL_BYTES EXPECTED_MD5 [N_PARTS]
set -u
URL="$1"; OUT="$2"; TOTAL="$3"; MD5="$4"; N="${5:-6}"
START=$( [ -f "$OUT" ] && stat -f %z "$OUT" || echo 0 )
REMAIN=$(( TOTAL - START ))
echo "already have $START bytes; fetching $REMAIN bytes in $N parts"
CHUNK=$(( (REMAIN + N - 1) / N ))
pids=()
for i in $(seq 0 $((N-1))); do
  s=$(( START + i*CHUNK )); e=$(( s + CHUNK - 1 )); [ $e -ge $((TOTAL-1)) ] && e=$((TOTAL-1))
  [ $s -gt $e ] && continue
  (
    part="$OUT.part$i"; want=$(( e - s + 1 ))
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
      have=$( [ -f "$part" ] && stat -f %z "$part" || echo 0 )
      [ "$have" -ge "$want" ] && break
      curl -sS --max-time 7200 -r $(( s + have ))-$e "$URL" >> "$part" || sleep 10
    done
    echo "part $i done: $(stat -f %z "$part") of $want bytes"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
for i in $(seq 0 $((N-1))); do
  part="$OUT.part$i"; [ -f "$part" ] || continue
  cat "$part" >> "$OUT" && rm "$part"
done
SIZE=$(stat -f %z "$OUT"); GOT=$(md5 -q "$OUT")
echo "final size $SIZE (expected $TOTAL); md5 $GOT (expected $MD5)"
[ "$SIZE" = "$TOTAL" ] && [ "$GOT" = "$MD5" ] && echo "VERIFIED OK" || echo "VERIFY FAILED"
