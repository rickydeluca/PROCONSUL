# PROCONSUL (PRObabilistic exploration of CONnectivity Significance patterns for disease modULe discovery)
PROCONSUL is an algorithm born by a modification of DIAMOnD by Ghiassian et all. (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004120).

DIAMOnD relies on the observation that CS is the best predictive quality for a disease associated protein.
Through the calculation and ranking of the CS of all proteins connected to known seed proteins, it is possible to evaluate which protein has more connections than expected to seed proteins.
The protein showing the lowest p-value (i.e., the greatest significance) is considered a most likely putative disease protein, and it is added to the seed protein set for another iteration to discover the next most likely putative protein.

However, by selecting among all the candidates exclusively the node with the lowest p-value, the risk is to cut out possible “putative protein paths” that appear less relevant at the beginning, but which can later lead to a greater performance.

For this reason we implemented a method for the probabilistic exploration of connectivity significance patterns in the tool PROCONSUL using the different p-values calculated for each candidate node as a reference point to create a probability distribution, and choosing the next putative disease protein according to this distribution. In this way, the CS still remains an important metric on which to base our predictions, but we also explore heuristically other potentially interesting paths that could lead to higher performances.

The code of PROCONSUL was adapted from the DIAMOnD code (https://github.com/dinaghiassian/DIAMOnD)

## Prerequisites


## How to run
to run PROCONSUL you can use:

```
python3 proconsul.py --network_file --seed_file --n --alpha(optional) --outfile_name(optional) --n_rounds(optional) --temp(optional) --top_p(optional) --top_k(optional)
```
Where:

| Argument 	| What it does 	|
|---	|---	|
| network_file 	| The edgelist must be provided as any delimiter-separated table. Make sure the delimiter does not exit in gene IDs and is consistent across the file. The first two columns of the table will be interpreted as an interaction gene1 <==> gene2. |
| seed_file  	| Table containing the seed genes (if table contains more than one column they must be tab-separated;the first column will be used only). |
| n 	| desired number of genes to predict, 200 is a reasonable starting point. |
| alpha 	| An integer representing weight of the seeds. (default: 1) |
| outfile_name 	| Results will be saved under this file name. (default: "proconsul_n_predicted_genes_temp_t(top_p_top_k_).txt") |
| n_runds 	| How many different rounds PROCONSUL will do to reduce statistical fluctuation. (default: 10) 	|
| temp 	| Temperature value for the PROCONSUL softmax function. (default: 1.0) 	|
| top_p 	| Probability threshold value for PROCONSUL nucleus sampling. If 0 no nucleus sampling. (default: 0.0) 	|
| top_k 	| Length of the pvalues subset for Top-K sampling. If 0 no top-k sampling. (default: 0)	|

For example: 
```
python3 proconsul.py --network_file "Example/PPI.txt" --seed_file "Example/seed_genes.txt" --n 200 --n_rounds 10 --temp 0.5
```
will run PROCONSUL using the network construced from the protein-protein interactions in "Example/PPI.txt" and starting from the seed genes in "Example/seed_genes.txt".
It will predict 200 new putative disease genes, the temperature for the softmax function will be 0.5 and to reduce the stastical fluctuations it will do 10 rounds.
