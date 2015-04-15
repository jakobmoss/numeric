set term pdfcairo
set out "out.A.pdf"
set key left
plot "A.txt" index 0 title "data" \
, "A.txt" index 1 title "lin" with lines \
, "A.txt" index 2 title "quad" with lines
