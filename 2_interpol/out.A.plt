set term pdfcairo
set out "out.A.pdf"
set key left
plot "out.A.txt" index 0 title "data" \
, "out.A.txt" index 1 title "lin" with lines \
, "out.A.txt" index 2 title "quad" with lines
