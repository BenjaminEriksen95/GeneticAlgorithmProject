requirements:
	pip3 install -r requirements.txt

onemax:
	bash scripts/onemax.sh

leadingones:
	bash scripts/leadingones.sh

jumpm:
	bash scripts/jumpm.sh

3sat:
	bash scripts/3sat_gen.sh

3sat_lib:
	bash scripts/3sat_lib.sh

tsp_lib:
	bash scripts/tsp_lib.sh

tsp:
	bash scripts/tsp_gen.sh

sorting:
	bash scripts/sorting.sh

swap_sorting:
	bash scripts/sorting.sh

plots:
	bash scripts/plots.sh

clean_logs:
	rm -r code/logs/*
