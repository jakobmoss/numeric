set term pdfcairo
set out "plot.A.pdf"
set yrange [-1.1:1.1]
set xzeroaxis lt 1 dt 3 lw 2 lc "black"
set key bottom left
plot "out.A.txt" index 0 title "y0" \
, "out.A.txt" index 1 title "y1" \
