set term pdfcairo
set out "plot.B.pdf"
set key left
plot "out.B.txt" index 0 with errorbars title "data"
