set term pdfcairo dashed
set out "plot.B.pdf"
set key left
plot "out.B.txt" index 0 notitle with lines lt 0 \
, "out.B.txt" index 1 title "data" \
, "out.B.txt" index 2 title "qspline" with lines \
, "out.B.txt" index 6 title "analytic deriv" with lines \
, "out.B.txt" index 3 title "spline deriv" with lines \
, "out.B.txt" index 5 title "analytic integ" with lines \
, "out.B.txt" index 4 title "spline integ" with lines
