cut -f1 papers >./paper_ids
grep -oE '<title>[^<]*</title>' papers >./paper_title
