set term pdfcairo
set out "plot.B.pdf"
set key left
plot "out.B.txt" index 0 with errorbars title "data" \
, "out.B.txt" index 1 with lines title "fit" \
, "out.B.txt" index 2 with lines title "fit + err" \
, "out.B.txt" index 3 with lines title "fit - err"
