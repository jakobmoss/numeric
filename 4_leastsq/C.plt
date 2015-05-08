set term pdfcairo
set out "plot.C.pdf"
set key left
plot "out.C.txt" index 0 with errorbars title "data" \
, "out.C.txt" index 1 using 1:2 w l title "Linear: y(x)" \
, "out.C.txt" index 1 using 1:3 w lines title "y(x) + dy" \
, "out.C.txt" index 1 using 1:4 w lines title "y(x) - dy"
