set term pdfcairo
set out "plot.B.pdf"
set xrange [0:12.5]
set yrange [-1.1:1.1]
set xzeroaxis lt 1 dt 3 lw 2 lc "black"
set key right
set title "y'' = -y ; RK23 = Embedded ; RK3 = Non-embedded"
set xlabel 'x'
set ylabel 'y'
plot "out.B.txt" index 0 title "RK23; y(0) = 0" \
, "out.B.txt" index 1 title "RK23; y(0) = 1" \
, "out.B.txt" index 2 title "RK3; y(0) = 0" \
, "out.B.txt" index 3 title "RK3; y(0) = 1"

set title "Number of function calls"
unset xzeroaxis
set xrange [0:11]
set yrange [0:1400]
set xlabel 'x'
set ylabel '# of calls'
set key left
plot "out.B.txt" index 0 using 1:3 with lines title "RK23; Embedded" \
, "out.B.txt" index 2 using 1:3 with lines title "RK3; Non-embedded" \
, "out.B.txt" index 4 using 1:3 with lines title "RK12; Embedded" \
