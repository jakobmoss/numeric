set term pdfcairo
set out "plot.A.pdf"
set title "Error estimate"
set logscale xy
set xrange [1:2e8]
set xlabel 'N'
set ylabel 'Error'
f(x) = a / sqrt(x)
fit f(x) "Aerr.dat" via a
plot "Aerr.dat" title "Data" \
, f(x) title "Fit: a/sqrt(x)"
