set term pdfcairo
set out "plot.C.pdf"
set key left
plot "out.C.txt" index 0 with errorbars title "data" \
, "out.C.txt" index 1 with lines title "Linear fit" \
