set term pdfcairo
set out "plot.A.pdf"
set yrange [-1.1:1.1]
set xzeroaxis lt 1 dt 3 lw 2 lc "black"
set key bottom left
set title "y'' = -y ; RK23"
set xlabel 'x'
set ylabel 'y'
plot "out.A.txt" index 0 title "y(0) = 0" \
, "out.A.txt" index 1 title "y(0) = 1" \
, sin(x), cos(x)
